"""
CAR-MFL with MNIST Dataset
Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning

MNIST handwritten digits with natural text descriptions.
- Image: 28x28 grayscale handwritten digit
- Text: Natural description like "this is digit five" or "the number is three"
- 10 clients: 6 multimodal, 3 image-only, 1 text-only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torchvision import datasets, transforms

# Set random seeds for reproducibility
torch.manual_seed(52)
np.random.seed(52)

# ============================================================================
# 1. VOCABULARY AND TEXT ENCODING
# ============================================================================

# Simple vocabulary for natural text descriptions
VOCAB = {
    '<PAD>': 0,
    'this': 1,
    'is': 2,
    'digit': 3,
    'the': 4,
    'number': 5,
    'zero': 6,
    'one': 7,
    'two': 8,
    'three': 9,
    'four': 10,
    'five': 11,
    'six': 12,
    'seven': 13,
    'eight': 14,
    'nine': 15,
    'a': 16,
    'handwritten': 17,
    'image': 18,
    'of': 19,
}

DIGIT_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def generate_text_description(label):
    """
    Generate a natural text description for a digit.

    Examples:
    - Label 0 -> "this is digit zero"
    - Label 5 -> "the number is five"
    - Label 8 -> "handwritten digit eight"

    Args:
        label: Digit class (0-9)

    Returns:
        text: Tensor of word indices (padded to length 10)
    """
    # Create varied natural descriptions
    templates = [
        ['this', 'is', 'digit', DIGIT_WORDS[label]],
        ['the', 'number', 'is', DIGIT_WORDS[label]],
        ['handwritten', 'digit', DIGIT_WORDS[label]],
        ['this', 'is', 'a', DIGIT_WORDS[label]],
        ['image', 'of', 'digit', DIGIT_WORDS[label]],
    ]

    # Select template based on label for consistency
    template = templates[label % len(templates)]

    # Convert words to indices
    indices = [VOCAB[word] for word in template]

    # Pad to fixed length (10 tokens)
    while len(indices) < 10:
        indices.append(VOCAB['<PAD>'])

    return torch.tensor(indices[:10], dtype=torch.long)


# ============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_mnist():
    """
    Load MNIST and create all datasets (public, train, test).

    Returns:
        public_data: 200 samples with both modalities
        client_train_data: 1000 samples for 10 clients (100 each)
        test_data: 1000 samples with both modalities
    """
    print("Loading MNIST dataset...")

    # Simple transform: normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full MNIST training set
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Load MNIST test set
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print(f"  MNIST train: {len(mnist_train)} samples")
    print(f"  MNIST test: {len(mnist_test)} samples")

    # Convert to multimodal format: (image, text, label)
    def to_multimodal(dataset, num_samples):
        data = []
        indices = torch.randperm(len(dataset))[:num_samples]
        for idx in indices:
            image, label = dataset[idx]
            text = generate_text_description(label)
            data.append((image, text, label))
        return data

    # Create datasets
    public_data = to_multimodal(mnist_train, 300)        # Larger public dataset
    client_train_data = to_multimodal(mnist_train, 500)  # For 10 clients (50 each)
    test_data = to_multimodal(mnist_test, 1000)          # Test set

    return public_data, client_train_data, test_data


def create_client_data(client_train_data, num_clients=10, samples_per_client=50):
    """
    Split data among clients with different modality configurations.

    Args:
        client_train_data: Full training data
        num_clients: Total number of clients (default: 10)
        samples_per_client: Samples per client (default: 100)

    Returns:
        client_data: List of client datasets
        client_modalities: List of modality types
    """
    client_data = []
    client_modalities = []

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        data_slice = client_train_data[start_idx:end_idx]

        # First 6 clients: multimodal (both image and text)
        if i < 6:
            client_data.append(data_slice)
            client_modalities.append('both')

        # Clients 6-8: image-only (3 clients)
        elif i < 9:
            # Remove text modality (set to None)
            image_only = [(img, None, lbl) for img, txt, lbl in data_slice]
            client_data.append(image_only)
            client_modalities.append('image')

        # Client 9: text-only (1 client)
        else:
            # Remove image modality (set to None)
            text_only = [(None, txt, lbl) for img, txt, lbl in data_slice]
            client_data.append(text_only)
            client_modalities.append('text')

    return client_data, client_modalities


# ============================================================================
# 3. MULTIMODAL MODEL
# ============================================================================

class MNISTMultimodalModel(nn.Module):
    """
    Multimodal model for MNIST images + natural text descriptions.

    Image: 28x28 grayscale -> CNN -> 128-dim
    Text: sequence of 10 words -> Embedding + pooling -> 128-dim
    Fusion: Concatenate -> Classifier -> 10 classes
    """

    def __init__(self, num_classes=10, vocab_size=20, embedding_dim=32):
        super().__init__()

        # Image encoder: CNN for MNIST (28x28 grayscale)
        self.image_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 28x28
        self.image_pool1 = nn.MaxPool2d(2, 2)  # 14x14
        self.image_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 14x14
        self.image_pool2 = nn.MaxPool2d(2, 2)  # 7x7
        self.image_fc = nn.Linear(32 * 7 * 7, 128)

        # Text encoder: Embedding + mean pooling (vocab size = 20)
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.text_fc = nn.Linear(embedding_dim, 128)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Classifier (concatenated features: 128 + 128 = 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def encode_image(self, image):
        """Encode image to 128-dim feature vector"""
        x = F.relu(self.image_conv1(image))
        x = self.image_pool1(x)
        x = F.relu(self.image_conv2(x))
        x = self.image_pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.image_fc(x)
        return x

    def encode_text(self, text):
        """Encode text to 128-dim feature vector"""
        x = self.text_embedding(text)  # (batch, seq_len, embedding_dim)
        x = torch.mean(x, dim=1)  # Mean pooling over sequence
        x = self.text_fc(x)
        return x

    def forward(self, image=None, text=None):
        """
        Forward pass. Expects at least one modality.

        Args:
            image: (batch, 1, 28, 28) or None
            text: (batch, 10) or None

        Returns:
            logits: (batch, 10)
        """
        features = []

        if image is not None:
            img_feat = self.encode_image(image)
            features.append(img_feat)
        else:
            # Zero-fill for missing image
            features.append(torch.zeros(text.size(0), 128))

        if text is not None:
            text_feat = self.encode_text(text)
            features.append(text_feat)
        else:
            # Zero-fill for missing text
            features.append(torch.zeros(image.size(0), 128))

        # Concatenate and classify
        fused = torch.cat(features, dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits


# ============================================================================
# 4. CROSS-MODAL RETRIEVAL (CORE OF CAR-MFL)
# ============================================================================

def label_to_set(label):
    """
    Convert a digit label (0-9) to a set representation for Jaccard similarity.

    For MNIST, we represent each digit as a set of its properties:
    - The digit itself
    - Whether it's even or odd
    - Its value range (0-2, 3-5, 6-9)

    Args:
        label: Digit label (0-9)

    Returns:
        Set of properties
    """
    properties = {label}  # The digit itself

    # Add even/odd property
    if label % 2 == 0:
        properties.add('even')
    else:
        properties.add('odd')

    # Add range property
    if label <= 2:
        properties.add('low')
    elif label <= 5:
        properties.add('mid')
    else:
        properties.add('high')

    return properties


def jaccard_similarity_labels(label1, label2):
    """
    Compute Jaccard similarity between two labels based on their properties.

    Jaccard similarity = |intersection| / |union|

    Args:
        label1: First label (0-9)
        label2: Second label (0-9)

    Returns:
        Jaccard similarity score (float between 0 and 1)
    """
    set1 = label_to_set(label1)
    set2 = label_to_set(label2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def retrieve_missing_modality(query_data, query_label, public_data, model, modality_type, top_k=5):
    """
    Retrieve the complementary modality from public data using CAR-MFL approach.

    This is the CORRECT CAR-MFL implementation:
    1. Find top-k nearest neighbors based on feature distance (image or text embedding)
    2. Among top-k, use Jaccard similarity on LABELS to find best match
    3. Return complementary modality from that best match

    Args:
        query_data: Single image or text tensor
        query_label: Label of the query sample (0-9)
        public_data: List of (image, text, label) tuples
        model: Current model for encoding
        modality_type: 'image' or 'text' (what we HAVE)
        top_k: Number of nearest neighbors to consider

    Returns:
        Retrieved complementary modality
    """
    model.eval()
    with torch.no_grad():
        # Step 1: Encode the query and find top-k by feature distance
        if modality_type == 'image':
            query_feat = model.encode_image(query_data.unsqueeze(0)).squeeze(0)
        else:  # text
            query_feat = model.encode_text(query_data.unsqueeze(0)).squeeze(0)

        # Compute distances to all public samples
        distances = []
        for pub_img, pub_text, pub_label in public_data:
            if modality_type == 'image':
                pub_feat = model.encode_image(pub_img.unsqueeze(0)).squeeze(0)
            else:
                pub_feat = model.encode_text(pub_text.unsqueeze(0)).squeeze(0)

            # L2 distance in feature space
            dist = torch.norm(query_feat - pub_feat).item()
            distances.append((dist, pub_img, pub_text, pub_label))

        # Sort by distance (closest first) and get top-k
        distances.sort(key=lambda x: x[0])
        top_k_candidates = distances[:min(top_k, len(distances))]

        # Step 2: Among top-k, use Jaccard similarity on labels to find best match
        best_jaccard = -1
        best_match = None

        for _, pub_img, pub_text, pub_label in top_k_candidates:
            jaccard_sim = jaccard_similarity_labels(query_label, pub_label)
            if jaccard_sim > best_jaccard:
                best_jaccard = jaccard_sim
                best_match = (pub_img, pub_text)

        # Step 3: Return the complementary modality
        if best_match is None:
            # Fallback to first candidate if Jaccard fails
            _, pub_img, pub_text, _ = top_k_candidates[0]
            best_match = (pub_img, pub_text)

        if modality_type == 'image':
            return best_match[1]  # Return text
        else:
            return best_match[0]  # Return image


# ============================================================================
# 4. CLIENT CLASS
# ============================================================================

class Client:
    """Federated learning client"""

    def __init__(self, data, modality_type, client_id):
        self.data = data
        self.modality_type = modality_type
        self.client_id = client_id

    def train_local(self, global_model, public_data, epochs=3, use_retrieval=True, lr=0.001):
        """
        Train locally with optional retrieval-based augmentation.

        Args:
            global_model: Current global model
            public_data: Public dataset for retrieval
            epochs: Number of local training epochs
            use_retrieval: If True, use CAR-MFL; else zero-fill
            lr: Learning rate

        Returns:
            Trained local model state dict
        """
        local_model = deepcopy(global_model)
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for image, text, label in self.data:
                # Augment if unimodal using CAR-MFL retrieval
                # Step 1: Find top-k by feature distance
                # Step 2: Use Jaccard similarity on labels to select best match
                if self.modality_type == 'image' and use_retrieval:
                    text = retrieve_missing_modality(image, label, public_data, local_model, 'image')
                elif self.modality_type == 'text' and use_retrieval:
                    image = retrieve_missing_modality(text, label, public_data, local_model, 'text')

                # Prepare batch
                if image is not None:
                    image = image.unsqueeze(0)
                if text is not None:
                    text = text.unsqueeze(0)
                label_tensor = torch.tensor([label])

                # Forward pass
                optimizer.zero_grad()
                logits = local_model(image, text)
                loss = criterion(logits, label_tensor)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pred = torch.argmax(logits, dim=1).item()
                correct += (pred == label)
                total += 1

            # Print client progress (commented out for cleaner output)
            # if epoch == epochs - 1:
            #     acc = 100 * correct / total
            #     print(f"  Client {self.client_id} ({self.modality_type}): Local Acc = {acc:.1f}%")

        return local_model.state_dict()


# ============================================================================
# 5. FEDERATED AVERAGING
# ============================================================================

def federated_average(global_model, client_weights):
    """Simple FedAvg: Average all client weights"""
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        stacked = torch.stack([client_weights[i][key].float() for i in range(len(client_weights))])
        global_dict[key] = torch.mean(stacked, dim=0)

    global_model.load_state_dict(global_dict)


# ============================================================================
# 6. EVALUATION
# ============================================================================

def evaluate(model, test_data):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for image, text, label in test_data:
            if image is not None:
                image = image.unsqueeze(0)
            if text is not None:
                text = text.unsqueeze(0)

            logits = model(image, text)
            pred = torch.argmax(logits, dim=1).item()

            if pred == label:
                correct += 1
            total += 1

    return 100 * correct / total


# ============================================================================
# 7. MAIN TRAINING
# ============================================================================

def main():
    print("=" * 80)
    print("CAR-MFL with MNIST Dataset")
    print("Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning")
    print("=" * 80)
    print()
    print("Dataset: MNIST handwritten digits (0-9)")
    print("Image modality: 28x28 grayscale images")
    print("Text modality: Natural descriptions (e.g., 'this is digit five')")
    print()

    # Hyperparameters
    NUM_CLIENTS = 10
    SAMPLES_PER_CLIENT = 50  # Reduced to make data scarcity more pronounced
    NUM_ROUNDS = 8
    LOCAL_EPOCHS = 2

    # Load and prepare all datasets
    public_data, client_train_data, test_data = load_and_prepare_mnist()

    # Create clients with different modality configurations
    client_data, client_modalities = create_client_data(
        client_train_data,
        num_clients=NUM_CLIENTS,
        samples_per_client=SAMPLES_PER_CLIENT
    )

    clients = [Client(client_data[i], client_modalities[i], i) for i in range(NUM_CLIENTS)]

    print(f"\nDataset Summary:")
    print(f"  Public data: {len(public_data)} samples (both modalities)")
    print(f"  Client training data: {len(client_train_data)} samples total")
    print(f"  Test data: {len(test_data)} samples (both modalities)")
    print()
    print(f"Client Configuration (10 clients):")
    print(f"  - Clients 0-5: Multimodal (image + text) - 6 clients")
    print(f"  - Clients 6-8: Image-only - 3 clients")
    print(f"  - Client 9: Text-only - 1 client")
    print(f"  - Samples per client: {SAMPLES_PER_CLIENT}")
    print()
    print(f"Training Configuration:")
    print(f"  - Federated rounds: {NUM_ROUNDS}")
    print(f"  - Local epochs per round: {LOCAL_EPOCHS}")
    print()
    print("Text representations:")
    print("  - Digit 0: 'this is digit zero'")
    print("  - Digit 1: 'the number is one'")
    print("  - Digit 2: 'handwritten digit two'")
    print("  - Digit 3: 'this is a three'")
    print("  - Digit 4: 'image of digit four'")
    print("  - (pattern repeats for digits 5-9)")
    print()

    # ========================================================================
    # BASELINE: Zero-filling
    # ========================================================================
    print("=" * 80)
    print("BASELINE: Training with Zero-Filling")
    print("(Missing modalities are filled with zeros)")
    print("=" * 80)

    baseline_model = MNISTMultimodalModel()

    for round_num in range(NUM_ROUNDS):
        client_weights = []

        for client in clients:
            weights = client.train_local(baseline_model, public_data, epochs=LOCAL_EPOCHS, use_retrieval=False)
            client_weights.append(weights)

        federated_average(baseline_model, client_weights)

        accuracy = evaluate(baseline_model, test_data)
        print(f"Round {round_num + 1}/{NUM_ROUNDS}: Accuracy = {accuracy:.2f}%")

    baseline_accuracy = evaluate(baseline_model, test_data)
    print(f"\n>>> Final Baseline Accuracy: {baseline_accuracy:.2f}%")
    print()

    # ========================================================================
    # CAR-MFL: Retrieval-based augmentation
    # ========================================================================
    print("=" * 80)
    print("CAR-MFL: Training with Retrieval-Based Augmentation")
    print("(Missing modalities retrieved from public dataset)")
    print("Method: Top-k nearest neighbors + Jaccard similarity")
    print("=" * 80)

    car_mfl_model = MNISTMultimodalModel()

    for round_num in range(NUM_ROUNDS):
        client_weights = []

        for client in clients:
            weights = client.train_local(car_mfl_model, public_data, epochs=LOCAL_EPOCHS, use_retrieval=True)
            client_weights.append(weights)

        federated_average(car_mfl_model, client_weights)

        accuracy = evaluate(car_mfl_model, test_data)
        print(f"Round {round_num + 1}/{NUM_ROUNDS}: Accuracy = {accuracy:.2f}%")

    car_mfl_accuracy = evaluate(car_mfl_model, test_data)
    print(f"\n>>> Final CAR-MFL Accuracy: {car_mfl_accuracy:.2f}%")
    print()

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    print(f"Baseline (zero-filling):      {baseline_accuracy:.2f}%")
    print(f"CAR-MFL (retrieval):          {car_mfl_accuracy:.2f}%")
    print(f"Improvement:                  +{car_mfl_accuracy - baseline_accuracy:.2f}%")
    print()

    if car_mfl_accuracy > baseline_accuracy:
        print("✓ CAR-MFL OUTPERFORMS BASELINE!")
        print("  Retrieval-based augmentation successfully improves over zero-filling")
        print("  The model learned to retrieve semantically similar samples from")
        print("  the public dataset to fill in missing modalities.")
    else:
        print("⚠ Results may vary due to random initialization.")
        print("  Try running multiple times.")

    print("=" * 80)


if __name__ == "__main__":
    main()
