"""
CAR-MFL with Real Chest X-ray Dataset + Medical Reports
Uses 500 real X-ray images with actual radiology reports
10 clients: 6 multimodal, 3 image-only, 1 text-only
Simple and clean implementation
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

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_xray_dataset(data_dir, num_samples=200):
    """Load X-ray images with real medical reports"""
    print("Loading chest X-ray dataset with medical reports...")

    # Load CSVs
    reports_df = pd.read_csv(os.path.join(data_dir, 'first_500_reports.csv'))
    proj_df = pd.read_csv(os.path.join(data_dir, 'first_500_projections.csv'))

    # Image transform
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    data = []
    img_dir = os.path.join(data_dir, 'first_500_images')

    # Use frontal images only
    frontal_df = proj_df[proj_df['projection'] == 'Frontal'].head(num_samples)

    for _, row in frontal_df.iterrows():
        uid = row['uid']
        filename = row['filename']

        # Get report for this uid
        report_row = reports_df[reports_df['uid'] == uid]
        if len(report_row) == 0:
            continue

        # Determine label: normal vs abnormal
        problems = str(report_row.iloc[0]['Problems']).lower()
        label = 0 if problems == 'normal' else 1

        # Load image
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)

            # Create text representation from problems/findings
            findings = str(report_row.iloc[0]['findings'])
            text = text_to_tensor(findings)

            data.append((img, text, label))
        except:
            pass

    print(f"  Loaded {len(data)} X-ray images with reports")
    print(f"  Normal: {sum(1 for _, _, l in data if l == 0)}")
    print(f"  Abnormal: {sum(1 for _, _, l in data if l == 1)}")

    np.random.shuffle(data)
    return data


def text_to_tensor(text):
    """Convert medical report text to tensor (simple word hashing)"""
    # Simple approach: use hash of words mod vocab size
    words = str(text).lower().split()[:20]  # First 20 words
    indices = [hash(w) % 100 for w in words]  # Vocab size = 100

    # Pad to length 20
    while len(indices) < 20:
        indices.append(99)  # PAD token

    return torch.tensor(indices[:20], dtype=torch.long)


# ============================================================================
# 2. SIMPLE MODEL
# ============================================================================

class XrayModel(nn.Module):
    """Simple multimodal model"""

    def __init__(self):
        super().__init__()

        # Image encoder
        self.img_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.img_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.img_fc = nn.Linear(32 * 16 * 16, 64)

        # Text encoder
        self.text_emb = nn.Embedding(100, 32)  # Vocab = 100
        self.text_fc = nn.Linear(32, 64)

        # Classifier
        self.fc = nn.Linear(128, 2)
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
# 3. RETRIEVAL
# ============================================================================

def retrieve_missing_modality(query_data, query_label, public_data, model, modality_type):
    """Retrieve complementary modality from public data"""
    model.eval()
    with torch.no_grad():
        if modality_type == 'image':
            q_feat = model.encode_image(query_data.unsqueeze(0)).squeeze()
        else:
            q_feat = model.encode_text(query_data.unsqueeze(0)).squeeze()

        best_dist = float('inf')
        best_item = None

        for pub_img, pub_text, pub_label in public_data:
            if modality_type == 'image':
                pub_feat = model.encode_image(pub_img.unsqueeze(0)).squeeze()
            else:
                pub_feat = model.encode_text(pub_text.unsqueeze(0)).squeeze()

            dist = torch.norm(q_feat - pub_feat).item()

            # Prefer same label
            if pub_label == query_label and dist < best_dist:
                best_dist = dist
                best_item = (pub_img, pub_text)

        if best_item is None:
            best_item = (public_data[0][0], public_data[0][1])

        return best_item[1] if modality_type == 'image' else best_item[0]


# ============================================================================
# 4. CLIENT & TRAINING
# ============================================================================

class Client:
    def __init__(self, data, modality_type):
        self.data = data
        self.modality_type = modality_type

    def train_local(self, model, public_data, use_retrieval=True):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for _ in range(2):  # 2 local epochs
            for img, txt, lbl in self.data:
                # Augment if needed
                if self.modality_type == 'image' and use_retrieval:
                    txt = retrieve_missing_modality(img, lbl, public_data, model, 'image')
                elif self.modality_type == 'text' and use_retrieval:
                    img = retrieve_missing_modality(txt, lbl, public_data, model, 'text')

                # Train
                img = img.unsqueeze(0) if img is not None else None
                txt = txt.unsqueeze(0) if txt is not None else None
                lbl = torch.tensor([lbl])

                opt.zero_grad()
                out = model(img, txt)
                loss = criterion(out, lbl)
                loss.backward()
                opt.step()

        return model.state_dict()


def federated_average(global_model, client_weights):
    """FedAvg"""
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        stacked = torch.stack([w[key].float() for w in client_weights])
        global_dict[key] = torch.mean(stacked, 0)
    global_model.load_state_dict(global_dict)


def evaluate(model, test_data):
    """Evaluate"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, txt, lbl in test_data:
            out = model(img.unsqueeze(0), txt.unsqueeze(0))
            if out.argmax(1).item() == lbl:
                correct += 1
    return 100 * correct / len(test_data)


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CAR-MFL with Real Chest X-ray Dataset + Medical Reports")
    print("=" * 70)
    print()

    # Load data
    data_dir = './data/chest_xray_with_report_500_images'
    all_data = load_xray_dataset(data_dir, num_samples=180)
    print()

    # Split
    public_data = all_data[:40]
    client_data_all = all_data[40:160]  # 120 for 10 clients
    test_data = all_data[160:]

    print(f"Data split:")
    print(f"  Public: {len(public_data)}")
    print(f"  Clients: {len(client_data_all)} (10 clients × 12 samples)")
    print(f"  Test: {len(test_data)}")
    print()

    # Create clients
    clients = []
    for i in range(10):
        start = i * 12
        data_slice = client_data_all[start:start+12]

        if i < 6:  # Multimodal
            clients.append(Client(data_slice, 'both'))
        elif i < 9:  # Image-only
            data_slice = [(img, None, lbl) for img, txt, lbl in data_slice]
            clients.append(Client(data_slice, 'image'))
        else:  # Text-only
            data_slice = [(None, txt, lbl) for img, txt, lbl in data_slice]
            clients.append(Client(data_slice, 'text'))

    print("Clients: 6 multimodal, 3 image-only, 1 text-only")
    print()

    # BASELINE
    print("-" * 70)
    print("BASELINE (Zero-Filling)")
    print("-" * 70)

    baseline = XrayModel()
    for r in range(10):
        weights = [c.train_local(deepcopy(baseline), public_data, use_retrieval=False) for c in clients]
        federated_average(baseline, weights)
        print(f"Round {r}: {evaluate(baseline, test_data):.1f}%")

    baseline_acc = evaluate(baseline, test_data)
    print(f"\nFinal Baseline: {baseline_acc:.1f}%")
    print()

    # CAR-MFL
    print("-" * 70)
    print("CAR-MFL (Retrieval)")
    print("-" * 70)

    carmfl = XrayModel()
    for r in range(10):
        weights = [c.train_local(deepcopy(carmfl), public_data, use_retrieval=True) for c in clients]
        federated_average(carmfl, weights)
        print(f"Round {r}: {evaluate(carmfl, test_data):.1f}%")

    carmfl_acc = evaluate(carmfl, test_data)
    print(f"\nFinal CAR-MFL: {carmfl_acc:.1f}%")
    print()

    # RESULTS
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline (zero-filling):  {baseline_acc:.1f}%")
    print(f"CAR-MFL (retrieval):      {carmfl_acc:.1f}%")
    print(f"Improvement:              +{carmfl_acc - baseline_acc:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
