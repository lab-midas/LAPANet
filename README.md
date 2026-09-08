# LAPANet

**LAPANet: Local-All-Pass Attention Network for Non-Rigid Image Registration in k-Space**

Official implementation of **LAPANet**, a deep learning framework for non-rigid motion estimation directly from accelerated MRI k-space data.

> 🚧 **Code coming soon.**
> The paper has been accepted for publication in **Medical Image Analysis**. The source code, pretrained models, and instructions will be released shortly.

---

## Overview

Image registration is traditionally performed in the image domain. However, highly accelerated MRI acquisitions can introduce severe undersampling and aliasing artifacts after reconstruction, which can substantially degrade the accuracy of image-based motion estimation.

**LAPANet** addresses this problem by performing non-rigid registration **directly in the acquired k-space**, avoiding the need to reconstruct an image solely for motion estimation.

The method combines the **Local-All-Pass (LAP)** formulation of non-rigid motion with a multi-scale attention-based neural network to estimate dense motion fields from complex-valued, multi-coil k-space data.

LAPANet is designed for highly time-resolved MRI applications where motion needs to be estimated from only a small number of acquired k-space samples per frame.

### Key features

* **Direct k-space registration** without requiring image reconstruction
* **Non-rigid motion estimation** using the Local-All-Pass formulation
* **Self-supervised training**, without requiring ground-truth deformation fields
* Supports **multi-coil complex-valued MRI data**
* Multi-scale feature extraction for capturing both local and global motion
* Applicable to **Cartesian and radial sampling trajectories**
* Designed for **highly accelerated and time-resolved MRI**
* Demonstrated for **cardiac and respiratory motion estimation**

---

## Method

LAPANet receives fixed and moving k-space data and predicts a dense non-rigid motion field describing the transformation between them.

The underlying Local-All-Pass formulation models non-rigid deformation as a sequence of local translational transformations. In the Fourier domain, local translations can be represented through phase modulations, providing a natural way to estimate motion directly from k-space.

The network operates on the real and imaginary components of the coil-resolved k-space data and uses a multi-resolution architecture to progressively refine the estimated motion.

The architecture consists of:

* **Global Residual Modules** for multi-scale k-space feature extraction
* **Encoder/Decoder blocks** for learning local and global representations
* **Attention mechanisms** for modeling long-range spatial dependencies
* **Motion Attention Modules** for progressively refining motion estimates
* A global translation component to account for overall image displacement

The model is trained in a self-supervised manner using the relationship between the fixed and moving acquisitions, avoiding the need for manually annotated deformation fields.

---

## Why k-Space Registration?

For highly accelerated MRI, reconstructing undersampled data can result in aliasing artifacts that interfere with conventional image-domain registration.

By operating directly on the acquired Fourier measurements, LAPANet can estimate motion before image reconstruction and therefore avoids relying on an intermediate aliased image.

This is particularly useful for applications requiring extremely high temporal resolution, including:

* Real-time cardiac MRI
* Respiratory motion estimation
* Motion-compensated reconstruction
* Motion tracking and characterization
* MR-guided radiotherapy
* Image-based gating and synchronization
* Real-time interventional MRI

---

## Results

LAPANet was evaluated on cardiac and respiratory motion estimation using both fully sampled and highly accelerated MRI acquisitions.

The method demonstrated robust motion estimation across different sampling trajectories and acceleration factors, including highly undersampled Cartesian and radial acquisitions. In reported experiments, LAPANet maintained reliable motion estimation with only a few k-space lines or spokes acquired per frame.

For cardiac motion, the method was demonstrated with temporal resolutions below **5 ms** under highly accelerated acquisitions, including experiments using as few as **2 Cartesian k-space lines per frame** and **3 radial spokes per frame**.

These results highlight the potential of direct k-space registration for high-frame-rate and real-time MRI applications.

---

## Code Availability

The implementation is currently being prepared for public release.

The repository will include:

* [ ] LAPANet model implementation
* [ ] Training scripts
* [ ] Inference scripts
* [ ] Pretrained model weights
* [ ] Data preprocessing utilities
* [ ] Example datasets / data preparation instructions
* [ ] Evaluation scripts
* [ ] Configuration files
* [ ] Reproduction instructions

**The code will be made publicly available shortly.**

---

## Citation

If you find LAPANet useful in your research, please cite our work:

```bibtex
@article{ghoul2024highly,
  title={Highly efficient non-rigid registration in k-space with application to cardiac magnetic resonance imaging},
  author={Ghoul, Aya and Hammernik, Kerstin and Lingg, Andreas and Krumm, Patrick and Rueckert, Daniel and Gatidis, Sergios and K{\"u}stner, Thomas},
  journal={arXiv preprint arXiv:2410.18834},
  year={2024}
}
```

> **Note:** The final bibliographic information will be updated once the journal publication is available.

---

## Acknowledgements

This work was developed as part of research on accelerated and motion-resolved magnetic resonance imaging.

---

## Status

**Paper:** Accepted at *Medical Image Analysis*
**Code:** Coming soon
**Pretrained models:** Coming soon

⭐ **Please watch/star this repository to be notified when the implementation is released.**


