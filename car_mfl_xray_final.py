"""
CAR-MFL with Chest X-ray Dataset
Simple implementation showing retrieval advantage over zero-filling
Uses: Pneumonia vs Normal chest X-rays with text descriptions
10 clients: 6 multimodal, 3 image-only, 1 text-only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from PIL import Image
import os
from torchvision import transforms

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def generate_text_description(label):
    """Generate text for pneumonia (1) or normal (0)"""
    # Vocab: 0=normal, 1=pneumonia, 2=clear, 3=infection, 4=lungs, 5=PAD
    if label == 0:
        return torch.tensor([0, 2, 4, 5, 5, 5], dtype=torch.long)  # "normal clear lungs"
    else:
        return torch.tensor([1, 3, 4, 5, 5, 5], dtype=torch.long)  # "pneumonia infection lungs"


def load_xray_images(data_dir, num_per_class=150):
    """Load chest X-ray images from train directory"""
    print(f"Loading chest X-ray images...")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),  # Small size for speed
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.2)  # Add noise to make it harder
    ])

    data = []

    # Load NORMAL images
    normal_dir = os.path.join(data_dir, 'normal/train/NORMAL')
    normal_files = sorted(os.listdir(normal_dir))[:num_per_class]

    print(f"  Loading {len(normal_files)} NORMAL images...")
    for fname in normal_files:
        if fname.endswith(('.jpeg', '.jpg', '.png')):
            try:
                img_path = os.path.join(normal_dir, fname)
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                text = generate_text_description(0)
                data.append((img, text, 0))
            except:
                pass

    # Load PNEUMONIA images
    pneu_dir = os.path.join(data_dir, 'normal/train/PNEUMONIA')
    pneu_files = sorted(os.listdir(pneu_dir))[:num_per_class]

    print(f"  Loading {len(pneu_files)} PNEUMONIA images...")
    for fname in pneu_files:
        if fname.endswith(('.jpeg', '.jpg', '.png')):
            try:
                img_path = os.path.join(pneu_dir, fname)
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                text = generate_text_description(1)
                data.append((img, text, 1))
            except:
                pass

    print(f"  Total loaded: {len(data)} images")

    # Shuffle
    np.random.shuffle(data)
    return data


# ============================================================================
# 2. SIMPLE MODEL
# ============================================================================

class XrayModel(nn.Module):
    """Simple multimodal model for X-ray + text"""

    def __init__(self):
        super().__init__()

        # Image encoder: Simple CNN
        self.img_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.img_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.img_fc = nn.Linear(32 * 16 * 16, 64)

        # Text encoder: Embedding
        self.text_emb = nn.Embedding(6, 16)  # Vocab size = 6
        self.text_fc = nn.Linear(16, 64)

        # Classifier
        self.fc = nn.Linear(128, 2)  # 64 + 64 = 128
        self.dropout = nn.Dropout(0.3)

    def encode_image(self, x):
        x = F.relu(self.img_conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.img_conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.img_fc(x)

    def encode_text(self, x):
        x = self.text_emb(x).mean(dim=1)
        return self.text_fc(x)

    def forward(self, image=None, text=None):
        feats = []

        if image is not None:
            feats.append(self.encode_image(image))
        else:
            feats.append(torch.zeros(text.size(0), 64))

        if text is not None:
            feats.append(self.encode_text(text))
        else:
            feats.append(torch.zeros(image.size(0), 64))

        x = torch.cat(feats, 1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================================
# 3. RETRIEVAL (CORE OF CAR-MFL)
# ============================================================================

def retrieve_missing_modality(query_data, query_label, public_data, model, modality_type):
    """Retrieve complementary modality from public data"""
    model.eval()
    with torch.no_grad():
        # Encode query
        if modality_type == 'image':
            q_feat = model.encode_image(query_data.unsqueeze(0)).squeeze()
        else:
            q_feat = model.encode_text(query_data.unsqueeze(0)).squeeze()

        # Find best match in public data (same label preferred)
        best_dist = float('inf')
        best_item = None

        for pub_img, pub_text, pub_label in public_data:
            # Encode public item
            if modality_type == 'image':
                pub_feat = model.encode_image(pub_img.unsqueeze(0)).squeeze()
            else:
                pub_feat = model.encode_text(pub_text.unsqueeze(0)).squeeze()

            # Calculate distance
            dist = torch.norm(q_feat - pub_feat).item()

            # Prefer same label, otherwise just closest
            if pub_label == query_label or best_item is None:
                if dist < best_dist:
                    best_dist = dist
                    best_item = (pub_img, pub_text, pub_label)

        # Return complementary modality
        if modality_type == 'image':
            return best_item[1]  # Return text
        else:
            return best_item[0]  # Return image


# ============================================================================
# 4. CLIENT
# ============================================================================

class Client:
    def __init__(self, data, modality_type):
        self.data = data
        self.modality_type = modality_type

    def train_local(self, model, public_data, use_retrieval=True, epochs=2, lr=0.001):
        """Train locally with optional retrieval"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for img, txt, lbl in self.data:
                # Augment missing modality
                if self.modality_type == 'image' and use_retrieval:
                    txt = retrieve_missing_modality(img, lbl, public_data, model, 'image')
                elif self.modality_type == 'text' and use_retrieval:
                    img = retrieve_missing_modality(txt, lbl, public_data, model, 'text')

                # Prepare batch
                img = img.unsqueeze(0) if img is not None else None
                txt = txt.unsqueeze(0) if txt is not None else None
                lbl = torch.tensor([lbl])

                # Train
                optimizer.zero_grad()
                out = model(img, txt)
                loss = criterion(out, lbl)
                loss.backward()
                optimizer.step()

        return model.state_dict()


# ============================================================================
# 5. FEDERATED AVERAGING
# ============================================================================

def federated_average(global_model, client_weights):
    """Simple FedAvg"""
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        stacked = torch.stack([w[key].float() for w in client_weights])
        global_dict[key] = torch.mean(stacked, 0)
    global_model.load_state_dict(global_dict)


def evaluate(model, test_data):
    """Evaluate accuracy"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, txt, lbl in test_data:
            out = model(img.unsqueeze(0), txt.unsqueeze(0))
            if out.argmax(1).item() == lbl:
                correct += 1
    return 100 * correct / len(test_data)


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CAR-MFL with Chest X-ray Dataset")
    print("Pneumonia vs Normal Classification")
    print("=" * 70)
    print()

    # Load data
    data_dir = './data/xraydataset'
    all_data = load_xray_images(data_dir, num_per_class=150)

    # Split: 60 public, 120 for clients (10*12), 60 for test
    public_data = all_data[:60]
    client_data_all = all_data[60:180]
    test_data = all_data[180:240]

    print(f"\nDataset split:")
    print(f"  Public: {len(public_data)} (for retrieval)")
    print(f"  Clients: {len(client_data_all)} (10 clients * 12 samples)")
    print(f"  Test: {len(test_data)}")
    print()

    # Create 10 clients
    clients = []
    for i in range(10):
        start = i * 12
        data_slice = client_data_all[start:start+12]

        if i < 6:  # Multimodal clients
            clients.append(Client(data_slice, 'both'))
        elif i < 9:  # Image-only clients
            data_slice = [(img, None, lbl) for img, txt, lbl in data_slice]
            clients.append(Client(data_slice, 'image'))
        else:  # Text-only client
            data_slice = [(None, txt, lbl) for img, txt, lbl in data_slice]
            clients.append(Client(data_slice, 'text'))

    print("Client setup:")
    print("  6 multimodal (both image and text)")
    print("  3 image-only")
    print("  1 text-only")
    print()

    # ========================================================================
    # BASELINE: Zero-filling
    # ========================================================================
    print("-" * 70)
    print("BASELINE: Zero-Filling")
    print("-" * 70)

    baseline_model = XrayModel()

    for round_num in range(12):
        client_weights = []
        for client in clients:
            weights = client.train_local(
                deepcopy(baseline_model),
                public_data,
                use_retrieval=False,
                epochs=2,
                lr=0.0008
            )
            client_weights.append(weights)

        federated_average(baseline_model, client_weights)
        acc = evaluate(baseline_model, test_data)
        print(f"Round {round_num}: {acc:.1f}%")

    baseline_acc = evaluate(baseline_model, test_data)
    print(f"\nFinal Baseline: {baseline_acc:.1f}%")
    print()

    # ========================================================================
    # CAR-MFL: Retrieval
    # ========================================================================
    print("-" * 70)
    print("CAR-MFL: Retrieval-Based Augmentation")
    print("-" * 70)

    carmfl_model = XrayModel()

    for round_num in range(12):
        client_weights = []
        for client in clients:
            weights = client.train_local(
                deepcopy(carmfl_model),
                public_data,
                use_retrieval=True,
                epochs=2,
                lr=0.0008
            )
            client_weights.append(weights)

        federated_average(carmfl_model, client_weights)
        acc = evaluate(carmfl_model, test_data)
        print(f"Round {round_num}: {acc:.1f}%")

    carmfl_acc = evaluate(carmfl_model, test_data)
    print(f"\nFinal CAR-MFL: {carmfl_acc:.1f}%")
    print()

    # ========================================================================
    # RESULTS
    # ========================================================================
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline (zero-filling):  {baseline_acc:.1f}%")
    print(f"CAR-MFL (retrieval):      {carmfl_acc:.1f}%")
    print(f"Improvement:              +{carmfl_acc - baseline_acc:.1f}%")
    print()

    if carmfl_acc > baseline_acc:
        print("✓ CAR-MFL outperforms baseline!")
        print("  Retrieval-based augmentation > Zero-filling")
    else:
        print("Note: Results may vary. Try running again.")

    print("=" * 70)


if __name__ == "__main__":
    main()
