# LAPANet

**LAPANet: Local-All-Pass Attention Network for Non-Rigid Image Registration in k-Space**

Official implementation of **LAPANet**, a deep learning framework for non-rigid
motion estimation directly from accelerated MRI k-space data.

08.09.2026: Accepted for publication in [**Medical Image Analysis**](https://doi.org/10.1016/j.media.2026.104296)

> 🚧 **Code release in progress.**
> This repository currently contains the model and training pipeline. Data
> preparation instructions, evaluation scripts, configuration files and
> final pretrained weights will be added shortly.

---

## Overview

Image registration is traditionally performed in the image domain. However,
highly accelerated MRI acquisitions introduce severe undersampling and aliasing
artifacts after reconstruction, which can substantially degrade the accuracy of
image-based motion estimation.

**LAPANet** addresses this problem by performing non-rigid registration
**directly in the acquired k-space**, avoiding the need to reconstruct an image
solely for the purpose of motion estimation.

The method realizes the **Local-All-Pass (LAP)** formulation of non-rigid motion
with a multi-scale attention-based neural network that estimates dense motion
fields from complex-valued, multi-coil k-space data. LAPANet is designed for
highly time-resolved MRI applications in which motion must be estimated from
only a small number of acquired k-space samples per frame.

### Key features

- **Direct k-space registration** without requiring image reconstruction
- **Non-rigid motion estimation** based on the Local-All-Pass formulation
- **Self-supervised training**, without requiring ground-truth deformation fields
- Supports **multi-coil complex-valued MRI data**
- Multi-scale feature extraction capturing both local and global motion
- Applicable to **Cartesian and radial sampling trajectories**
- Designed for **highly accelerated and time-resolved MRI**
- Demonstrated for **cardiac and respiratory motion estimation**

---

## Method

LAPANet receives fixed and moving k-space data and predicts a dense non-rigid
motion field describing the transformation between them.

The underlying Local-All-Pass formulation models non-rigid deformation as a
sequence of local translational transformations. In the Fourier domain, local
translations can be represented through phase modulations, providing a natural
way to estimate motion directly from k-space.

The network operates on the real and imaginary components of the coil-resolved
k-space data and uses a multi-resolution architecture to progressively refine
the estimated motion. Its main components are:

- **Global Residual Modules** for multi-scale k-space feature extraction
- **Encoder/Decoder blocks** for learning local and global representations
- **Attention mechanisms** for modeling long-range spatial dependencies
- **Motion Attention Modules** for progressively refining motion estimates
- A **global translation component** to account for overall image displacement

The model is trained in a self-supervised manner using the relationship between
the fixed and moving acquisitions, avoiding the need for manually annotated
deformation fields.

---

## Why k-Space Registration?

For highly accelerated MRI, reconstructing undersampled data can result in
aliasing artifacts that interfere with conventional image-domain registration.

By operating directly on the acquired Fourier measurements, LAPANet can estimate
motion *before* image reconstruction and therefore avoids relying on an
intermediate aliased image. This is particularly useful for applications that
require extremely high temporal resolution, including:

- Real-time cardiac MRI
- Respiratory motion estimation
- Motion-compensated reconstruction
- Motion tracking and characterization
- MR-guided radiotherapy
- Image-based gating and synchronization
- Real-time interventional MRI

---

## Results

LAPANet was evaluated on cardiac and respiratory motion estimation using both
fully sampled and highly accelerated MRI acquisitions.

The results reported in the paper were obtained on **in-house acquired data**.
Acceleration was simulated retrospectively using two sampling strategies:

- **VISTA** variable-density Cartesian sampling
- **Radial golden-angle** sampling

The method demonstrated robust motion estimation across different sampling
trajectories and acceleration factors. In reported experiments, LAPANet
maintained reliable motion estimation with only a few k-space lines or spokes
acquired per frame.

For cardiac motion, the method was demonstrated with temporal resolutions below
**5 ms** under highly accelerated acquisitions, including experiments using as
few as **2 Cartesian k-space lines per frame** and **3 radial spokes per frame**.

These results highlight the potential of direct k-space registration for
high-frame-rate and real-time MRI applications.

---

## Data

The in-house datasets used in the paper (VISTA-accelerated Cartesian and
retrospectively radial-golden-angle-accelerated cine MRI) **cannot be made
publicly available** due to institutional data-sharing restrictions.

To make the training pipeline reproducible without relying on our internal data,
this repository includes a data loader for the publicly available
[**CMRxRecon 2023 challenge dataset**](https://cmrxrecon.github.io/Home.html), which
uses the same multi-coil, time-resolved cine MRI format. CMRxRecon provides
**Cartesian** cine acquisitions at multiple acceleration factors (4×, 8×, 10×)
together with fully sampled references. This allows the k-space registration
pipeline to be trained and evaluated end-to-end on a public benchmark.

The CMRxRecon loader is intended as a **working example** that mirrors the
structure of the in-house pipeline. Users with their own k-space data can adapt
the loader to their own acquisition format.

### Expected directory layout (CMRxRecon example)

```<data_root_dir>/
TrainingSet/
AccFactor04/ P001/cine_sax.mat, cine_lax.mat
AccFactor08/ P001/cine_sax.mat, cine_lax.mat
AccFactor10/ P001/cine_sax.mat, cine_lax.mat
FullSample/ P001/cine_sax.mat, cine_lax.mat
```

Each `.mat` file is a MATLAB v7.3 (HDF5) file containing a single k-space
variable (`kspace_sub04`, `kspace_sub08`, `kspace_sub10`, or `kspace_full`).
The loader reads the fully sampled reference from `FullSample/` and the
sub-sampled k-space from the corresponding `AccFactorXX/` folder of the same
subject.

---

## Getting Started

### Environment Setup

```bash
conda env create -f environment.yml
conda activate <env-name>
pip install -r requirements.txt
```

### Requirements

- PyTorch (CUDA recommended)
- [`merlin`](https://github.com/midas-tum/merlin/tree/master) for VISTA sampling utilities

### Training on CMRxRecon (example)

```bash
# Using default YAML values
python path/to/project/src/scripts/run_cmrxrecon.py

# Pointing to a custom YAML file
python path/to/project/src/scripts/run_cmrxrecon.py --config /path/to/custom_config.yaml

# Overriding specific parameters
python path/to/project/src/scripts/run_cmrxrecon.py --data_dir /path/to/CMRxRecon/TrainingSet --batch_size 8
```

---

## How to use the HuggingFace App

Upload a **5D k-space file** with shape `(Frames, Slices, Coils, H, W)`, pick a
fixed and a moving frame, and get a color-coded cardiac motion field.

### 1. Prepare your input

The app expects complex-valued k-space in one of these formats:

| Extension | Layout | Notes |
|---|---|---|
| `.npy` / `.npz` | `(F, S, C, H, W)` | `np.complex64` or `np.complex128` |
| `.h5` / `.hdf5` | `(F, S, C, H, W)` | dataset named `kspace`, `kspace_full`, or similar |
| `.mat` | `(F, S, C, H, W)` | MATLAB v7.3 (HDF5) compound `real`/`imag` dtype |

- `F` — frames (time points)
- `S` — slices
- `C` — coils (must be **10**)
- `H`, `W` — spatial dimensions

The file details panel updates automatically after upload and shows you the
detected shape and axis sizes, so you can verify before running.

### 2. Pick a frame pair

Two pairs of indices define the two inputs:

| Control | Meaning | Typical value |
|---|---|---|
| `z1` | slice index of the **fixed** input | `0` |
| `t1` | frame index of the **fixed** input | `0` |
| `z2` | slice index of the **moving** input | `0` |
| `t2` | frame index of the **moving** input | `t1 + 1` |

All indices are **0-based**. For cardiac motion, use adjacent frames
(`t2 = t1 + 1`) in the same slice (`z1 = z2`). Cross-slice registration is
possible but represents a different problem.

### 3. Run and read the output

Click **Run Motion Estimation**. You get four views:

- **Inputs** — coil-combined magnitude images of the fixed and moving frames
  (`ifft2c` → sum over coils → `abs`, normalized to `[0, 1]`).
- **Flow · color** — the estimated motion field, HSV-encoded via
  [`flow_vis`](https://github.com/tomrunia/OpticalFlow_Visualization):
  hue = direction, saturation/brightness = magnitude.
- **Flow · quiver** — the same field as arrows on a black background.
  Tune *arrow spacing* and *arrow scale* under "Quiver settings".

Every panel can be downloaded as a PNG.

### 4. Tips and troubleshooting

- **Wrong axis order?** Verify with `np.shape(kspace)` locally — the app
  requires exactly 5 dimensions and will reject anything else.
- **Coil count mismatch?** The model was trained on 10 coils; pad or
  coil-compress your data before uploading.

---
## Inference on Jupyter

A step-by-step Jupyter notebook reproducing the whole pipeline, loading,
preprocessing, forward pass, and flow visualisation, is available at
[`notebooks/inference.ipynb`](https://github.com/lab-midas/LAPANet/blob/master/notebooks/inference.ipynb).
---

## Code Availability

The implementation is currently being prepared for public release.

The repository will include:

* [x] LAPANet model implementation
* [x] Training scripts
* [x] CMRxRecon data loader (public example)
* [x] Huggingface model weights
* [x] Evaluation scripts
* [x] Configuration files
* [ ] Huggingface space
* [ ] Final checkpoint upload

**The code will be completed shortly.**

---

## Citation

If you find LAPANet useful in your research, please cite our work:

```bibtex
@article{ghoul2026learning,
  title={Learning efficient non-rigid registration in k-space for accelerated Magnetic Resonance Imaging},
  author={Ghoul, Aya and Hammernik, Kerstin and Lingg, Andreas and Krumm, Patrick and Rueckert, Daniel and Gatidis, Sergios and K{\"u}stner, Thomas},
  journal={Medical Image Analysis},
  pages={104296},
  year={2026},
  publisher={Elsevier}
}
```

