# LAPANet: Local-All-Pass Attention Network for Non-Rigid Image Registration in k-Space

[![Paper](https://img.shields.io/badge/Paper-Medical%20Image%20Analysis-blue)](https://doi.org/10.1016/j.media.2026.104296)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-orange.svg)](https://pytorch.org)


---

## Overview

**LAPANet** is a deep learning framework for **non-rigid motion estimation directly from accelerated MRI k-space data**, bypassing image reconstruction. This approach enables accurate motion estimation at **sub-5 millisecond temporal resolution** with as few as **2 Cartesian k-space lines per frame** or **3 radial spokes per frame**, making it ideal for dynamic and real-time MRI applications.

### Why k-Space Registration?

Highly accelerated MRI reconstructions suffer from severe undersampling artifacts and aliasing that degrade image quality and disrupt feature matching. By operating directly on acquired Fourier measurements, LAPANet estimates motion **before image reconstruction**, avoiding reliance on aliased images and enabling reliable motion estimation under extreme acceleration.

### Key Advantages

| Aspect | LAPANet | Image-Based Methods             |
|--------|---------|---------------------------------|
| **Input** | Raw k-space (accelerated) | Reconstructed images (degraded) |
| **Reconstruction Needed** | ❌ No | ✅ Yes                           |
| **High Acceleration Robustness** | ✅ Yes | ❌ No                            |
| **Temporal Resolution** | <5 ms | >20-50 ms                       |

---

## Key Features

### Core Capabilities

- ✅ **Direct k-space registration** without image reconstruction
- ✅ **Non-rigid motion estimation** based on Local-All-Pass (LAP) formulation
- ✅ **Self-supervised training** without requiring annotated deformation fields
- ✅ **Multi-coil information** for complex-valued MRI data
- ✅ **Multi-scale architecture** capturing local and global motion patterns
- ✅ **Trajectory agnostic** — supports Cartesian and radial sampling
- ✅ **Highly accelerated** — validated at R=78 (Cartesian) and R=104 (radial)
- ✅ **Real-time capable** — ~30 ms inference per frame pair
- ✅ **Cardiac & respiratory** motion estimation validated

### Building Modules

- **Global Residual Modules** — Multi-scale k-space feature extraction at full resolution
- **Attention Mechanisms** — Long-range spatial dependency modeling
- **Motion Attention Modules** — Progressive refinement across scales
- **k-Space Magnitude Consistency Loss** — Global structural guidance
- **Efficient Architecture** — 4000× speedup vs. prior LAP-based methods

---

## Quick Start

### Installation 

```bash
# Clone repository
git clone https://github.com/lab-midas/LAPANet.git
cd LAPANet

# Create environment
conda env create -f environment.yml
conda activate lapanet

# Install dependencies
pip install -r requirements.txt
```

### Download Pretrained Model

```bash
# Download from HuggingFace (coming soon)
python scripts/download_model.py --model_name lapanet_cmrxrecon
```


### Run Jupyter Notebook (Interactive)

```bash
# Step-by-step inference with visualization
jupyter notebook notebooks/inference.ipynb
```

### Run the App (Interactive)

```bash
python hf_space/app.py
```

---

## Installation

### System Requirements

| Component | Requirement | Notes                        |
|-----------|-------------|------------------------------|
| **OS** | Linux/macOS/Windows | Tested on Ubuntu 20.04+      |
| **Python** | 3.8–3.11 | 3.8+ recommended             |
| **CUDA** | 11.0+ | Highly recommended for speed |
| **GPU Memory** | ≥8 GB | 16 GB+ for batch processing  |
| **RAM** | ≥16 GB | 32 GB recommended            |
| **Disk** | ≥50 GB | For datasets + checkpoints   |

### Step-by-Step Installation

#### 1. **Clone Repository**

```bash
git clone https://github.com/lab-midas/LAPANet.git
cd LAPANet
```

#### 2. **Create Conda Environment**

```bash
# Option A: Use provided environment (recommended)
conda env create -f environment.yml
conda activate lapanet

# Option B: Manual setup
conda create -n lapanet python=3.10
conda activate lapanet
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 3. **Install Dependencies**

```bash
# Install from requirements
pip install -r requirements.txt
```

#### 4. **Install MERLIN (Optional, for VISTA sampling utilities)**

```bash
# not needed for cmrxrecon
git clone https://github.com/midas-tum/merlin.git
cd merlin
pip install -e .
```

#### 5. **Verify Installation**

```bash
python -c "
import torch
import numpy as np
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

---

## Usage Guide

### 1. Data Preparation

#### Use Public CMRxRecon Dataset (Recommended)

```bash
# Download from CMRxRecon challenge website
# https://cmrxrecon.github.io/Home.html
# Expected directory structure:
# data/CMRxRecon/
# ├── TrainingSet/
# │   ├── AccFactor04/P001/cine_sax.mat
# │   ├── AccFactor08/P001/cine_sax.mat
# │   ├── AccFactor10/P001/cine_sax.mat
# │   └── FullSample/P001/cine_sax.mat
# └── ValidationSet/...

```


---

## Training

### Training from Scratch

#### 1. **Configure Training**

Edit `config/train_cmrxrecon.yaml`:

#### 2. **Launch Training**

```bash
# Single GPU
python scripts/run_cmrxrecon.py --config config/train_cmrxrecon.yaml

# Override config parameters
python scripts/run_cmrxrecon.py \
  --config configs/experiments/my_experiment.yaml \
  --batch_size 64 \
  --learning_rate 5e-5 \
  --num_epochs 100
```

#### 3. **Monitor Training**

```bash
# Using TensorBoard
tensorboard --logdir checkpoints/logs

# Using Weights & Biases (optional)
pip install wandb
# Set WANDB_API_KEY environment variable
```



---

## Data

### CMRxRecon Public Dataset

LAPANet is demonstrated in this repo using the **CMRxRecon 2023 Challenge** multi-coil cardiac cine dataset.

**Download:**
- Official website: https://cmrxrecon.github.io/Home.html

**Dataset Properties:**

| Property | Value |
|----------|-------|
| **Subjects** | 200 training + 100 test |
| **Sequence** | 2D bSSFP cine |
| **Coils** | 10 (multi-coil) |
| **Spatial Resolution** | 1.9 × 1.9 mm² |
| **Temporal Phases** | 25 frames |
| **Slice Thickness** | 8 mm |
| **Acceleration Factors** | 4×, 8×, 10× (Cartesian) |
| **Format** | MATLAB v7.3 (.mat files) |

**Expected Directory Structure:**

```
data/CMRxRecon/
├── TrainingSet/
│   ├── AccFactor04/
│   │   ├── P001/
│   │   │   ├── cine_sax.mat (undersampled k-space)
│   │   │   └── cine_lax.mat
│   │   ├── P002/...
│   │   └── ...
│   ├── AccFactor08/
│   ├── AccFactor10/
│   └── FullSample/
│       ├── P001/
│       │   ├── cine_sax.mat (fully sampled reference)
│       │   └── cine_lax.mat
│       └── ...
└── TestSet/
    ├── P201/ ... (similar structure)
```

### Custom Data Format

For your own data:

1. **Save as HDF5/MAT/NPY** with shape `(F, S, C, H, W)` (complex-valued)
2. Or **Create a custom loader**



### In-House Data (Non-Public)

The original in-house datasets used in the paper cannot be released due to ethical restrictions. 

---

## Citation

If you use LAPANet in your research, please cite:

```bibtex
@article{ghoul2026learning,
  title={Learning efficient non-rigid registration in k-space for accelerated Magnetic Resonance Imaging},
  author={Ghoul, Aya and Hammernik, Kerstin and Lingg, Andreas and Krumm, Patrick and Rueckert, Daniel and Gatidis, Sergios and K{\"u}stner, Thomas},
  journal={Medical Image Analysis},
  volume={115},
  pages={104296},
  year={2027},
  doi={10.1016/j.media.2026.104296},
  publisher={Elsevier}
}
```

### Paper Links

- **Published Article**: https://doi.org/10.1016/j.media.2026.104296
- **ArXiv Preprint**: https://arxiv.org/abs/2410.18834
- **GitHub Repository**: https://github.com/lab-midas/LAPANet
- **HuggingFace Repository**: https://github.com/lab-midas/LAPANet


---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## Support & Contact

- **Issues**: GitHub Issues tracker
- **Email**: aya.ghoul@med.uni-tuebingen.de
- **Lab Website**: https://www.midas.uni-tuebingen.de/

---

## Additional Resources

- [Model Architecture Explanation](docs/architecture.md) (coming soon)
- [Loss Functions Deep Dive](docs/losses.md) (coming soon)
- [Training Tips & Tricks](docs/training_guide.md) (coming soon)
- [Paper PDF](https://doi.org/10.1016/j.media.2026.104296)
- [Supplementary Materials](https://doi.org/10.1016/j.media.2026.104296) (on journal website)

---

**Last Updated**: September 2026  
**Current Version**: 1.0.0  
**Code Status**: 🟢 Actively Maintained