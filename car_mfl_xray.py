"""
CAR-MFL with Real Chest X-ray Dataset + Medical Reports
Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning

Uses real chest X-ray images (frontal view) with impression text from radiology reports
- Image: 64x64 grayscale chest X-ray
- Text: Clinical impression from radiologist report
- 10 clients: 6 multimodal, 3 image-only, 1 text-only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
import os
from torchvision import transforms

torch.manual_seed(55)
np.random.seed(55)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def text_to_tensor(text, max_length=30, vocab_size=500):
    """
    Convert medical report impression text to tensor using word hashing.

    Args:
        text: Impression text from radiologist report
        max_length: Maximum sequence length
        vocab_size: Size of vocabulary

    Returns:
        Tensor of word indices
    """
    # Clean and tokenize text
    text = str(text).lower().strip()
    words = text.split()[:max_length]

    # Hash words to indices (deterministic)
    indices = [abs(hash(word)) % (vocab_size - 1) + 1 for word in words]  # 1 to vocab_size-1

    # Pad to max_length with 0 (PAD token)
    while len(indices) < max_length:
        indices.append(0)

    return torch.tensor(indices[:max_length], dtype=torch.long)


def load_and_prepare_xray_data(data_dir, max_samples=400):
    """
    Load chest X-ray dataset with frontal images and impression text.

    Args:
        data_dir: Path to dataset directory
        max_samples: Maximum number of samples to load

    Returns:
        public_data: Samples for retrieval (both modalities)
        client_train_data: Samples for clients
        test_data: Test samples (both modalities)
    """
    print("Loading Chest X-ray dataset...")
    print("  Dataset: Frontal chest X-rays with radiologist impressions")

    # Load CSV files
    reports_df = pd.read_csv(os.path.join(data_dir, 'first_500_reports.csv'))
    proj_df = pd.read_csv(os.path.join(data_dir, 'first_500_projections.csv'))

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img_dir = os.path.join(data_dir, 'first_500_images')

    # Get frontal images only
    frontal_df = proj_df[proj_df['projection'] == 'Frontal']
    print(f"  Found {len(frontal_df)} frontal X-ray images")

    # Load all available data
    all_data = []

    for _, row in frontal_df.iterrows():
        uid = row['uid']
        filename = row['filename']

        # Get report for this uid
        report_row = reports_df[reports_df['uid'] == uid]
        if len(report_row) == 0:
            continue

        # Get impression text (primary clinical finding)
        impression = report_row.iloc[0]['impression']
        if pd.isna(impression) or str(impression).strip() == '':
            continue

        # Determine label: normal vs abnormal based on Problems field
        problems = str(report_row.iloc[0]['Problems']).lower()
        label = 0 if problems == 'normal' else 1

        # Load image
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)

            # Convert impression text to tensor
            text = text_to_tensor(impression)

            all_data.append((img, text, label))

            if len(all_data) >= max_samples:
                break

        except Exception as e:
            continue

    print(f"  Successfully loaded {len(all_data)} samples")

    # Count labels
    normal_count = sum(1 for _, _, l in all_data if l == 0)
    abnormal_count = sum(1 for _, _, l in all_data if l == 1)
    print(f"  Normal: {normal_count}, Abnormal: {abnormal_count}")

    # Shuffle data
    np.random.shuffle(all_data)

    # Split into public, train, and test
    # Public: 100 samples for retrieval
    # Train: 200 samples for 10 clients (20 each)
    # Test: 100 samples for evaluation
    public_data = all_data[:100]
    client_train_data = all_data[100:300]
    test_data = all_data[300:400]

    return public_data, client_train_data, test_data


def create_client_data(client_train_data, num_clients=10, samples_per_client=20):
    """
    Split data among clients with different modality configurations.

    Args:
        client_train_data: Full training data
        num_clients: Total number of clients (default: 10)
        samples_per_client: Samples per client (default: 20)

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
# 2. MULTIMODAL MODEL
# ============================================================================

class XrayModel(nn.Module):
    """
    Multimodal model for chest X-ray images + impression text.

    Image: 64x64 grayscale -> CNN -> 128-dim
    Text: sequence of 30 words -> Embedding + pooling -> 128-dim
    Fusion: Concatenate -> Classifier -> 2 classes (normal/abnormal)
    """

    def __init__(self, vocab_size=500, embedding_dim=64):
        super().__init__()

        # Image encoder: CNN for 64x64 grayscale X-rays
        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.img_fc = nn.Linear(32 * 16 * 16, 128)

        # Text encoder: Embedding + mean pooling
        self.text_emb = nn.Embedding(vocab_size, embedding_dim)
        self.text_fc = nn.Linear(embedding_dim, 128)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.dropout = nn.Dropout(0.3)

    def encode_image(self, x):
        """Encode X-ray image to 128-dim feature vector"""
        x = F.relu(self.img_conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.img_conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.img_fc(x)

    def encode_text(self, x):
        """Encode impression text to 128-dim feature vector"""
        x = self.text_emb(x)  # (batch, seq_len, embedding_dim)
        x = x.mean(dim=1)  # Mean pooling over sequence
        return self.text_fc(x)

    def forward(self, image=None, text=None):
        """
        Forward pass. Expects at least one modality.

        Args:
            image: (batch, 1, 64, 64) or None
            text: (batch, 30) or None

        Returns:
            logits: (batch, 2)
        """
        feats = []

        if image is not None:
            feats.append(self.encode_image(image))
        else:
            # Zero-fill for missing image
            feats.append(torch.zeros(text.size(0), 128))

        if text is not None:
            feats.append(self.encode_text(text))
        else:
            # Zero-fill for missing text
            feats.append(torch.zeros(image.size(0), 128))

        x = torch.cat(feats, 1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================================
# 3. CROSS-MODAL RETRIEVAL (CORE OF CAR-MFL)
# ============================================================================

def label_to_set(label):
    """
    Convert a binary label (0=normal, 1=abnormal) to a set representation for Jaccard similarity.

    For chest X-rays:
    - Normal (0): {0, 'normal', 'healthy'}
    - Abnormal (1): {1, 'abnormal', 'pathology'}

    Args:
        label: Binary label (0 or 1)

    Returns:
        Set of properties
    """
    if label == 0:
        return {0, 'normal', 'healthy'}
    else:
        return {1, 'abnormal', 'pathology'}


def jaccard_similarity_labels(label1, label2):
    """
    Compute Jaccard similarity between two labels based on their properties.

    Jaccard similarity = |intersection| / |union|

    Args:
        label1: First label (0 or 1)
        label2: Second label (0 or 1)

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
        query_label: Label of the query sample (0=normal, 1=abnormal)
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

    def train_local(self, global_model, public_data, epochs=2, use_retrieval=True, lr=0.001):
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
    print("CAR-MFL with Chest X-ray Dataset")
    print("Cross-Modal Augmentation by Retrieval for Multimodal Federated Learning")
    print("=" * 80)
    print()
    print("Dataset: Frontal chest X-rays with radiologist impressions")
    print("Image modality: 64x64 grayscale chest X-rays")
    print("Text modality: Clinical impression from radiology reports")
    print()

    # Hyperparameters
    NUM_CLIENTS = 10
    SAMPLES_PER_CLIENT = 20
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 2

    # Load and prepare datasets
    data_dir = './data/chest_xray_with_report_500_images'
    public_data, client_train_data, test_data = load_and_prepare_xray_data(data_dir, max_samples=400)
    print()

    # Create clients with different modality configurations
    client_data, client_modalities = create_client_data(
        client_train_data,
        num_clients=NUM_CLIENTS,
        samples_per_client=SAMPLES_PER_CLIENT
    )

    clients = [Client(client_data[i], client_modalities[i], i) for i in range(NUM_CLIENTS)]

    print(f"Dataset Summary:")
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

    # ========================================================================
    # BASELINE: Zero-filling
    # ========================================================================
    print("=" * 80)
    print("BASELINE: Training with Zero-Filling")
    print("(Missing modalities are filled with zeros)")
    print("=" * 80)

    baseline_model = XrayModel()

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

    car_mfl_model = XrayModel()

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
        print("⚠ Results may vary due to random initialization or data quality.")
        print("  Try running multiple times.")

    print("=" * 80)


if __name__ == "__main__":
    main()
