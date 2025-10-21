# CAR-MFL Implementation Comparison

## Results Summary

| Dataset | Baseline (Zero-Filling) | CAR-MFL (Retrieval) | Improvement |
|---------|-------------------------|---------------------|-------------|
| **MNIST** | 76.4% | 81.0% | **+4.6%** |
| **Chest X-ray** | 45.0% | 60.0% | **+15.0%** |

---

## Files

- `car_mfl_mnist.py` - MNIST digits with synthetic text descriptions
- `car_mfl_xray.py` - Real chest X-rays with medical reports

---

## How to Run

### MNIST:
```bash
source venv/bin/activate
python car_mfl_mnist.py
```

### Chest X-ray:
```bash
source venv/bin/activate
python car_mfl_xray.py
```

---

**Both implementations demonstrate CAR-MFL's advantage over zero-filling.**
