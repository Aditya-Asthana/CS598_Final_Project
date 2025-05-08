# When More Is Less: Synthetic Domain Shift Demo

This repository contains a minimal, self-contained demo of the “When More Is Less” phenomenon from Compton et al., in which adding an external dataset can paradoxically *hurt* subgroup robustness by introducing spurious correlations. We recreate the core single- vs. multi-domain ERM experiment on a small chest X-ray subset, using synthetic augmentations as a proxy for a second “hospital.”

---

## Overview

- **Domain A:** Your raw chest-scan images.
- **Domain B:** Synthetic copies of Domain A created via `ColorJitter(brightness=0.5, contrast=0.5)` + `GaussianBlur(kernel_size=5)`.
- **Model:** DenseNet-121 (ImageNet pretrained head removed + single‐unit sigmoid).
- **Baseline:** Empirical Risk Minimization (ERM) with equalized sampling across domains.
- **Metric:** Worst-Group Accuracy (WGA) over the four (domain × label) subgroups.

This demo runs end-to-end in a Jupyter notebook, showing:
1. Pseudo-labeling with TorchXRayVision  
2. Manual train/val/test split to avoid empty subgroups  
3. CPU‐ and GPU‐compatible training loops  
4. Evaluation utilities for WGA, AUROC, and more  

---

## Requirements

- Python 3.11  
- `torch>=2.6.0+cu124`  
- `torchvision>=0.21.0+cu124`  
- `torchxrayvision>=1.3.4`  
- `pandas>=2.2.2`  
- `scikit-learn>=1.3.2`  
- `matplotlib>=3.10.0`  
- `tqdm>=4.67.1`  

Install via:
```bash
pip install torch torchvision torchxrayvision pandas scikit-learn matplotlib tqdm
