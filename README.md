# LAPANet

**LAPANet: Local-All-Pass Attention Network for Non-Rigid Image Registration in k-Space**

Official implementation of **LAPANet**, a deep learning framework for non-rigid
motion estimation directly from accelerated MRI k-space data.

Accepted for publication in **Medical Image Analysis**.

> 🚧 **Code release in progress.**
> This repository currently contains the model and training pipeline. Data
> preparation instructions, evaluation scripts, configuration files and
> pretrained weights will be added shortly.

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
[**CMRxRecon 2023 challenge dataset**](https://www.smicmrxrecon.com/), which
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

### Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- `h5py`, `numpy`, `scipy`
- [`merlin`](https://github.com/midas-tum/merlin/tree/master) for VISTA sampling utilities

### Training on CMRxRecon (example)

```bash
python src/scripts/run_cmrxrecon.py \
    --data_dir /path/to/CMRxRecon/TrainingSet \
    --view sax \
    --R_list 4 8 10 \
    --out_shape 512 512 \
    --batch_size 16 \
    --num_workers 4
```

---

## Code Availability

The implementation is currently being prepared for public release.

The repository will include:

* [x] LAPANet model implementation
* [x] Training scripts
* [x] CMRxRecon data loader (public example)
* [ ] Evaluation scripts
* [ ] Configuration files
* [ ] Pretrained model weights

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

