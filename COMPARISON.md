# CAR-MFL Implementation Comparison

## Two Implementations Available

This repository contains **two implementations** of CAR-MFL, each serving different purposes:

---

## 1. **Synthetic Data Version** (`car_mfl_simple.py`)

### Purpose
**Teaching and demonstration** - Shows the CAR-MFL concept with dramatic, easy-to-see improvements.

### Characteristics
- ✅ **Synthetic data**: Random tensors (no downloads needed)
- ✅ **3 clients**: 1 image-only, 1 text-only, 1 multimodal
- ✅ **Large improvement**: +16% (48% → 64%)
- ✅ **Fast**: Runs in 30 seconds
- ✅ **Self-contained**: ~300 lines, single file
- ✅ **No dependencies**: Just PyTorch and NumPy

### Run
```bash
source venv/bin/activate
python car_mfl_simple.py
```

### Output
```
BASELINE: 48.0%
CAR-MFL:  64.0%
Improvement: +16.0%
```

### When to Use
- Quick demos and presentations
- Understanding the core concept
- Teaching federated learning
- Showing clear benefits of retrieval vs zero-filling

---

## 2. **MNIST Data Version** (`car_mfl_mnist.py`)

### Purpose
**Research and realism** - Uses real data with realistic improvements you'd see in practice.

### Characteristics
- ✅ **Real MNIST images**: Handwritten digits (28×28)
- ✅ **Generated text**: Digit descriptions (e.g., "curved, round")
- ✅ **10 clients**: 6 multimodal, 3 image-only, 1 text-only
- ✅ **Realistic improvement**: +0.2-0.5% (94.4% → 94.6%)
- ✅ **Extensible**: Easy to swap MNIST for other datasets
- ✅ **Downloads data**: First run downloads ~10MB

### Run
```bash
source venv/bin/activate
python car_mfl_mnist.py
```

### Output
```
BASELINE: 94.4%
CAR-MFL:  94.6%
Improvement: +0.2%
```

### When to Use
- Building on this for research
- Testing on real data
- Extending to other datasets (Fashion-MNIST, CIFAR-10)
- Showing realistic performance gains

---

## Side-by-Side Comparison

| Feature | Synthetic | MNIST |
|---------|-----------|-------|
| **Dataset** | Random tensors | Real MNIST images |
| **Text** | Random indices | Generated descriptions |
| **Image size** | 64×64×3 | 28×28×1 |
| **Classes** | 2 (binary) | 10 (digits) |
| **Clients** | 3 | 10 |
| **Unimodal clients** | 2 | 4 |
| **Multimodal clients** | 1 | 6 |
| **Public data** | 100 samples | 150 samples |
| **Per-client data** | 50 samples | 50 samples |
| **Baseline accuracy** | ~48% | ~94% |
| **CAR-MFL accuracy** | ~64% | ~95% |
| **Improvement** | +16% | +0.2% |
| **Runtime** | 30 sec | 1-2 min |
| **First run** | 30 sec | 2-3 min (download) |
| **File size** | ~320 lines | ~520 lines |
| **Dependencies** | torch, numpy | torch, torchvision, numpy |

---

## Why Different Improvements?

### Synthetic Version: Large Improvement (+16%)
The synthetic data is **intentionally difficult**:
- Very weak signals in each modality alone
- Only 20% correlation between features and labels
- Random noise dominates the signal
- Neither modality alone is sufficient

**Result**: Zero-filling fails badly (48%), while CAR-MFL's retrieval helps significantly (64%).

### MNIST Version: Small Improvement (+0.2%)
MNIST is **too easy**:
- Images alone achieve >90% accuracy
- Even with zero-filled text, the model learns well
- The task doesn't really need both modalities
- Ceiling effect at ~95% accuracy

**Result**: Both methods work well, but CAR-MFL is slightly better and converges faster.

---

## Which Should You Use?

### Use **Synthetic** (`car_mfl_simple.py`) if you want to:
- ✅ Understand CAR-MFL concept quickly
- ✅ See dramatic visual improvements
- ✅ Demo for presentations/teaching
- ✅ Avoid downloading datasets
- ✅ Run experiments quickly

### Use **MNIST** (`car_mfl_mnist.py`) if you want to:
- ✅ Work with real data
- ✅ Extend to other datasets
- ✅ See realistic improvements
- ✅ Build on this for research
- ✅ Test with more clients (10 instead of 3)

### Use **Both** if you want to:
- ✅ Compare synthetic vs real data behavior
- ✅ Show both conceptual clarity and practical utility
- ✅ Teach: start with synthetic, then show real data

---

## Core Mechanism (Same in Both)

Despite the different datasets, **both implementations use the same retrieval mechanism**:

```python
def retrieve_missing_modality(query_data, query_label, public_data, model, modality_type):
    """
    1. Encode the available modality (e.g., image)
    2. Find similar samples in public dataset
    3. Retrieve the complementary modality (e.g., text)
    4. Use it to create a complete sample
    """
```

This is the **key innovation** of CAR-MFL that both versions demonstrate.

---

## Extending Further

Both implementations can be extended:

### From Synthetic:
1. Add more classes (multi-class instead of binary)
2. Increase data size
3. Add more modalities (e.g., audio)
4. Test different retrieval strategies (top-k, weighted average)

### From MNIST:
1. **Fashion-MNIST**: Clothing items instead of digits
2. **CIFAR-10**: Color images (32×32×3)
3. **ChestX-ray**: Medical images (requires larger download)
4. **Better text**: Use real captions or descriptions
5. **Non-IID splits**: Different clients have different digit distributions

---

## Summary

| | Synthetic | MNIST |
|-|-----------|-------|
| **Purpose** | Teaching/Demo | Research/Realism |
| **Strength** | Clear improvements | Real data |
| **Weakness** | Artificial data | Small improvements |
| **Best for** | Understanding concepts | Building applications |

**Both are correct implementations of CAR-MFL** - they just serve different purposes!
