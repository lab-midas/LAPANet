#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Aya

Developed at the University Hospital of Tübingen.
Copyright © 2026 University Hospital of Tübingen.

If you'd like to use or share this code, please get in touch with
Aya Ghoul <aya.ghoul@med.uni-tuebingen.de>.

Loss functions for medical image registration (LAPANet).

This module defines loss criteria for training non-rigid and rigid flow-based
image registration models. It supports multi-scale pyramid losses and both
spatial and k-space (Fourier) domain computations.

Key concepts:
- Photometric loss: measures alignment quality by comparing warped and reference images
- Smooth loss: encourages smooth flow fields with boundary-aware weighting
- Multi-scale loss: applies losses at different image resolutions with weighted combination
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torchvision.transforms import Resize
from utils import warp_non_rigid_flow, warp_rigid_flow, fft2c

# ============================================================================
# Loss Constants
# ============================================================================
# Multi-scale pyramid loss weights: applied when flow_pred is a list of
# multi-resolution flow predictions. Lower resolution (earlier) predictions
# get less weight; higher resolution (later) predictions get more weight.
MULTI_SCALE_LOSS_WEIGHTS = [0.05, 0.15, 0.2, 0.6]

# K-space (Fourier domain) photometric loss weight relative to spatial domain
K_SPACE_PHOTOMETRIC_WEIGHT = 0.01

# Weight for rigid translation shift loss relative to other photometric losses
TRANSLATION_SHIFT_LOSS_WEIGHT = 0.5


# ============================================================================
# Base Criterion Class
# ============================================================================
class CriterionBase(nn.Module):
    """
    Base class for combining multiple loss terms.

    This factory class constructs individual loss functions from a configuration
    object and manages their weights for combined loss computation.

    Args:
        config: Configuration object with attributes:
            - which: List of loss names to include (e.g., ['photometric_weighted', 'smooth'])
            - loss_weights: List of weights for each loss term
            - Attributes for each loss (e.g., config.photometric_weighted, config.smooth)
    """

    def __init__(self, config):
        super(CriterionBase, self).__init__()
        self.loss_names = config.which
        self.loss_weights = config.loss_weights
        self.loss_list = []

        # Instantiate each loss term from config
        for loss_name in config.which:
            loss_args = self._get_loss_config(config, loss_name)
            loss_item = self.get_loss(loss_name=loss_name, args_dict=loss_args)
            self.loss_list.append(loss_item)

    def _get_loss_config(self, config, loss_name):
        """
        Safely extract loss-specific configuration.

        Args:
            config: Configuration object
            loss_name: Name of the loss (e.g., 'photometric_weighted')

        Returns:
            Dictionary of arguments for the loss constructor
        """
        loss_config = getattr(config, loss_name, None)
        if loss_config is None:
            raise ValueError(f"Config has no attribute '{loss_name}'")
        return loss_config.__dict__

    def get_loss(self, loss_name, args_dict):
        """
        Factory method to instantiate a loss function.

        Args:
            loss_name: Type of loss ('photometric_weighted' or 'smooth')
            args_dict: Keyword arguments for loss constructor

        Returns:
            Instantiated loss module

        Raises:
            NotImplementedError: If loss_name is not registered
        """
        if loss_name == 'photometric_weighted':
            return PhotometricLossWeighted(**args_dict)
        elif loss_name == 'smooth':
            return SmoothLoss(**args_dict)
        else:
            raise NotImplementedError(
                f"Loss '{loss_name}' not implemented. "
                f"Available losses: 'photometric_weighted', 'smooth'"
            )


# ============================================================================
# Main Multi-Scale Loss Function
# ============================================================================
class LAPANetLoss2D(CriterionBase, nn.Module):
    """
    Combined loss for 2D medical image registration.

    Computes weighted combination of photometric and smoothness losses.
    Handles both:
    - Single-scale: flow_pred is a single tensor
    - Multi-scale: flow_pred is a list of flow predictions at different resolutions

    The multi-scale case applies losses at each resolution with different weights,
    allowing the model to refine registration progressively from coarse to fine.

    Args:
        config: Configuration object (see CriterionBase)
    """

    def __init__(self, config):
        super().__init__(config=config)

    def forward(self, flow_pred=None, ref=None, mov=None, shift=None, box=None):
        """
        Compute combined loss for image registration.

        Args:
            flow_pred (Tensor or list): 
                - Single-scale (Tensor): Non-rigid flow field of shape (B, 2, H, W)
                - Multi-scale (list): List of flow tensors at different resolutions
            ref (Tensor): Reference image of shape (B, C, H, W)
            mov (Tensor): Moving image to be warped, shape (B, C, H, W)
            shift (Tensor or None): Rigid translation shift of shape (B, 2)
                If provided, applies additional rigid alignment loss
            box (Tensor or None): Spatial weighting mask of shape (B, 1, H, W)
                Used by PhotometricLossWeighted to mask loss computation

        Returns:
            dict: Loss dictionary with keys for each loss term plus 'total_loss'
                Example: {
                    'photometric_multi_coil': 0.32,
                    'photometric_single_coil': 0.18,
                    'translation_shift_loss': 0.05,
                    'photometric_shift_loss': 0.12,
                    'k_photometric_loss': 0.008,
                    'smooth': 0.15,
                    'total_loss': 0.825
                }
        """
        loss_dict = {}
        total_loss = 0.0

        # Iterate over each registered loss term with its weight
        for loss_name, loss_weight, loss_term in zip(
                self.loss_names, self.loss_weights, self.loss_list
        ):
            if loss_name == 'photometric_weighted':
                # Compute photometric loss (measures alignment quality)
                photometric_loss = self._compute_photometric_loss(
                    flow_pred=flow_pred,
                    ref=ref,
                    mov=mov,
                    shift=shift,
                    box=box,
                    loss_term=loss_term
                )
                loss_dict.update(photometric_loss)

                # Get scalar loss value (for multi-scale, sum weighted components)
                if isinstance(photometric_loss, dict):
                    i_loss = sum(
                        v for k, v in photometric_loss.items()
                        if isinstance(v, torch.Tensor)
                    )
                    i_loss = i_loss.mean() if i_loss.numel() > 1 else i_loss
                else:
                    i_loss = photometric_loss

            elif loss_name == 'smooth':
                # Compute smoothness loss (encourages smooth flow)
                i_loss = self._compute_smooth_loss(
                    flow_pred=flow_pred,
                    ref=ref,
                    loss_term=loss_term
                )

            else:
                raise KeyError(
                    f"Loss '{loss_name}' not registered in forward(). "
                    f"This should have been caught at initialization."
                )

            # Store individual loss and accumulate total
            loss_dict[loss_name] = i_loss
            total_loss += loss_weight * i_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict

    def _compute_photometric_loss(self, flow_pred, ref, mov, shift, box, loss_term):
        """
        Compute photometric loss (image alignment quality).

        Handles both single-scale and multi-scale flow predictions.
        For single-scale: computes losses on multi-coil, single-coil, translation, and k-space.
        For multi-scale: applies losses at each resolution with pyramid weighting.

        Args:
            flow_pred: Flow prediction (Tensor or list of Tensors)
            ref: Reference image
            mov: Moving image
            shift: Rigid translation (optional)
            box: Spatial weight mask (optional)
            loss_term: PhotometricLossWeighted module

        Returns:
            dict: Loss components
        """
        if torch.is_tensor(flow_pred):
            # Single-scale flow prediction case
            return self._compute_single_scale_photometric_loss(
                flow_pred=flow_pred,
                ref=ref,
                mov=mov,
                shift=shift,
                box=box,
                loss_term=loss_term
            )
        else:
            # Multi-scale flow prediction case (pyramid of resolutions)
            return self._compute_multi_scale_photometric_loss(
                flow_pred_list=flow_pred,
                ref=ref,
                mov=mov,
                box=box,
                loss_term=loss_term
            )

    def _compute_single_scale_photometric_loss(self, flow_pred, ref, mov, shift, box, loss_term):
        """
        Compute photometric loss for single-scale flow.

        Computes the following loss components:
        1. photometric_multi_coil: Raw multi-coil warped vs reference
        2. photometric_single_coil: Summed (single-coil) warped vs reference
        3. translation_shift_loss (if shift provided): Rigid alignment loss
        4. photometric_shift_loss (if shift provided): Combined rigid+non-rigid loss
        5. k_photometric_loss: Fourier domain photometric loss

        Args:
            flow_pred: Non-rigid flow field (B, 2, H, W)
            ref: Reference image (B, C, H, W)
            mov: Moving image (B, C, H, W)
            shift: Rigid translation (B, 2) or None
            box: Weight mask (B, 1, H, W) or None
            loss_term: PhotometricLossWeighted module

        Returns:
            dict: Individual loss components and combined photometric loss
        """
        mse = nn.MSELoss()
        loss_dict = {}

        # ====== Component 1: Multi-coil photometric loss ======
        # Warp moving image using non-rigid flow and compare to reference
        img_warped_multi_coil = warp_non_rigid_flow(mov, flow_pred)
        loss_multi_coil = loss_term(ref, img_warped_multi_coil, box)
        loss_dict['photometric_multi_coil'] = loss_multi_coil

        # ====== Component 2: Single-coil photometric loss ======
        # Sum across coils (channels) to get magnitude, then compute loss
        # This assumes summing channels produces a valid intensity image
        mov_sum = torch.sum(mov, dim=1, keepdim=True)  # (B, 1, H, W)
        ref_sum = torch.sum(ref, dim=1, keepdim=True)  # (B, 1, H, W)

        img_warped_single_coil = warp_non_rigid_flow(mov_sum, flow_pred)
        loss_single_coil = loss_term(ref_sum, img_warped_single_coil, box)
        loss_dict['photometric_single_coil'] = loss_single_coil

        # ====== Component 3 & 4: Rigid translation loss (if shift provided) ======
        photometric_shift_loss = torch.tensor(0.0, device=flow_pred.device)
        translation_shift_loss = torch.tensor(0.0, device=flow_pred.device)

        if torch.is_tensor(shift):
            # Rigid alignment: apply translation to already-warped single-coil image
            img_warped_trans_rigid = warp_rigid_flow(img_warped_single_coil, shift)
            translation_shift_loss = loss_term(ref_sum, img_warped_trans_rigid, box)
            loss_dict['translation_shift_loss'] = translation_shift_loss

            # Combined rigid + non-rigid: add translation to flow field then warp
            # This creates a flow field that includes both rigid and non-rigid components
            batch_size = flow_pred.shape[0]
            height = flow_pred.shape[2]
            width = flow_pred.shape[3]

            # Broadcast shift (B, 2) to flow shape (B, 2, H, W)
            shift_flow = shift.unsqueeze(2).unsqueeze(3)  # (B, 2, 1, 1)
            shift_flow = shift_flow.expand(batch_size, 2, height, width)  # (B, 2, H, W)

            # Combine: non-rigid flow + rigid translation
            combined_flow = shift_flow + flow_pred
            img_warped_combined = warp_non_rigid_flow(mov, combined_flow)
            photometric_shift_loss = loss_term(ref, img_warped_combined, box)
            loss_dict['photometric_shift_loss'] = photometric_shift_loss

        # ====== Component 5: K-space (Fourier) photometric loss ======
        # Transform images to frequency domain and compare magnitudes
        # This provides additional constraint in k-space, useful for MRI data
        k_ref = fft2c(ref)  # (B, C, H, W) in frequency domain
        k_warped = fft2c(img_warped_multi_coil)  # (B, C, H, W) in frequency domain

        # Compute magnitude spectra
        k_ref_magnitude = torch.abs(k_ref)
        k_warped_magnitude = torch.abs(k_warped)

        k_photometric_loss = mse(k_warped_magnitude, k_ref_magnitude)
        loss_dict['k_photometric_loss'] = k_photometric_loss

        # ====== Combine all photometric components ======
        # Weighted sum of all loss components
        # Note: coefficients (0.01, 0.5) control relative importance
        total_photometric_loss = (
                K_SPACE_PHOTOMETRIC_WEIGHT * k_photometric_loss +
                photometric_shift_loss +
                TRANSLATION_SHIFT_LOSS_WEIGHT * translation_shift_loss +
                loss_single_coil +
                loss_multi_coil
        )

        return total_photometric_loss

    def _compute_multi_scale_photometric_loss(self, flow_pred_list, ref, mov, box, loss_term):
        """
        Compute photometric loss for multi-scale (pyramid) flow predictions.

        Applies photometric loss at each resolution level with decreasing weights,
        allowing coarse-to-fine registration. Each flow prediction is at a different
        resolution, and reference/moving images are resized to match.

        Args:
            flow_pred_list: List of flow tensors at decreasing resolutions
            ref: Reference image (full resolution)
            mov: Moving image (full resolution)
            box: Weight mask (full resolution)
            loss_term: PhotometricLossWeighted module

        Returns:
            Weighted sum of losses across all scales
        """
        total_photometric_loss = torch.tensor(0.0, device=flow_pred_list[0].device)

        # Iterate over each scale with its weight
        for scale_idx, (flow_at_scale, weight) in enumerate(
                zip(flow_pred_list, MULTI_SCALE_LOSS_WEIGHTS)
        ):
            # Get flow resolution for this scale
            flow_height = flow_at_scale.shape[2]
            flow_width = flow_at_scale.shape[3]

            # Resize images and mask to match flow resolution
            # (flow may be at 1/2, 1/4, 1/8 of original resolution)
            ref_at_scale = Resize((flow_height, flow_width))(ref)
            mov_at_scale = Resize((flow_height, flow_width))(mov)

            if box is not None:
                box_at_scale = Resize((flow_height, flow_width))(box)
            else:
                box_at_scale = None

            # Warp moving image at this scale
            img_warped = warp_non_rigid_flow(mov_at_scale, flow_at_scale)

            # Compute loss at this scale and accumulate with weight
            scale_loss = loss_term(ref_at_scale, img_warped, box_at_scale)
            total_photometric_loss += weight * scale_loss

        return total_photometric_loss

    def _compute_smooth_loss(self, flow_pred, ref, loss_term):
        """
        Compute smoothness loss to encourage regular flow fields.

        Handles both single-scale and multi-scale flow predictions.

        Args:
            flow_pred: Flow prediction (Tensor or list of Tensors)
            ref: Reference image (used for boundary-aware weighting)
            loss_term: SmoothLoss module

        Returns:
            Scalar smoothness loss
        """
        if torch.is_tensor(flow_pred):
            # Single-scale: directly compute smooth loss
            return loss_term(flow_pred, ref)
        else:
            # Multi-scale: apply smooth loss at each resolution
            total_smooth_loss = torch.tensor(0.0, device=flow_pred[0].device)

            for scale_idx, (flow_at_scale, weight) in enumerate(
                    zip(flow_pred, MULTI_SCALE_LOSS_WEIGHTS)
            ):
                # Resize reference to match flow resolution
                flow_height = flow_at_scale.shape[2]
                flow_width = flow_at_scale.shape[3]
                ref_at_scale = Resize((flow_height, flow_width))(ref)

                # Compute and accumulate weighted smooth loss
                scale_smooth_loss = loss_term(flow_at_scale, ref_at_scale)
                total_smooth_loss += weight * scale_smooth_loss

            return total_smooth_loss


# ============================================================================
# Photometric Loss (Image Alignment Quality)
# ============================================================================
class PhotometricLossWeighted(nn.Module):
    """
    Weighted photometric loss for image alignment.

    Measures image registration quality by comparing reference and warped images.
    Uses a power-law weighting scheme (α parameter) and spatial weighting mask
    to handle non-uniform image quality or regions of interest.

    The loss is computed as:
        loss = (|I_ref - I_warped|^2)^α, weighted by a mask

    where α ∈ (0, 1) makes the loss less sensitive to large intensity differences
    (useful for noisy or artifacts-prone data).

    Args:
        alpha (float): Power parameter for robustness (default: 0.45)
            - α = 1.0: L2 loss (sensitive to outliers)
            - α = 0.5: Square root of L2 (moderate robustness)
            - α < 0.5: Even more robust (dampens large errors)
        eps (float): Small epsilon to avoid numerical issues (default: 1e-6)
    """

    def __init__(self, alpha=0.45, eps=1e-6):
        super(PhotometricLossWeighted, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, inputs, outputs, weight=None):
        """
        Compute weighted photometric loss.

        Args:
            inputs (Tensor): Reference image (B, C, H, W)
            outputs (Tensor): Warped moving image (B, C, H, W)
            weight (Tensor or None): Spatial weight mask (B, 1, H, W)
                If None, uniform weighting is used

        Returns:
            Scalar loss value
        """
        # Compute intensity difference
        diff = inputs - outputs

        # Compute squared magnitude (handle complex numbers via conjugate)
        # For real-valued inputs, conjugate has no effect
        squared_magnitude = torch.conj(diff) * diff

        # Apply power-law weighting for robustness
        loss = torch.pow(squared_magnitude + self.eps, exponent=self.alpha)

        # Apply spatial weighting if provided
        if weight is not None:
            # Multiply by weight and normalize by mean weight
            # This prevents the loss from being trivially reduced by zero-weighting
            loss = torch.multiply(weight, loss) / weight.mean()

        # Return scalar loss (average over all elements)
        return torch.mean(loss)


# ============================================================================
# Flow Field Gradient Computation
# ============================================================================
def gradient(data):
    """
    Compute spatial gradients of a 4D tensor (batch, channel, height, width).

    Computes forward differences along spatial dimensions:
    - D_dx: gradients along horizontal (width) direction
    - D_dy: gradients along vertical (height) direction

    Args:
        data (Tensor): Input tensor of shape (B, C, H, W)

    Returns:
        tuple: (D_dx, D_dy)
            - D_dx (Tensor): shape (B, C, H, W-1), horizontal gradients
            - D_dy (Tensor): shape (B, C, H-1, W), vertical gradients
    """
    # Horizontal gradient: difference along width (last dimension)
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]  # (B, C, H, W-1)

    # Vertical gradient: difference along height (third dimension)
    D_dy = data[:, :, 1:, :] - data[:, :, :-1, :]  # (B, C, H-1, W)

    return D_dx, D_dy


# ============================================================================
# Smoothness Loss (Flow Field Regularity)
# ============================================================================
class SmoothLoss(nn.Module):
    """
    Smoothness loss to regularize flow fields.

    Encourages smooth, spatially-coherent flow by penalizing large gradients.
    Uses boundary-aware weighting: reduces smoothness constraints at image edges
    where large flow gradients are expected (e.g., where object boundaries move).

    The loss penalizes both horizontal and vertical flow gradients:
        loss = sum(|∇_x flow| + |∇_y flow|) * boundary_weight

    where boundary_weight = exp(-α * |∇I|) for boundary-aware variant.
    This allows discontinuities in flow at detected image edges.

    Args:
        boundary_awareness (bool): Whether to use boundary-aware weighting
            If True, reduces penalty at locations with high image gradients
            If False, applies uniform smoothness penalty (default: True)
        alpha (float): Steepness of edge weighting (default: 10)
            Higher alpha → sharper edge detection, stricter boundary adherence
    """

    def __init__(self, boundary_awareness=True, alpha=10):
        super(SmoothLoss, self).__init__()
        self.boundary_awareness = boundary_awareness
        self.alpha = alpha
        # Use 1st-order gradient smoothing (could extend to 2nd order, Huber loss, etc.)
        self.func_smooth = self.smooth_grad_1st

    def smooth_grad_1st(self, flow, image):
        """
        Compute 1st-order gradient smoothness penalty.

        Computes magnitude of flow gradients and optionally weights them by
        inverse of image gradients to allow larger flow changes at edges.

        Args:
            flow (Tensor): Optical flow field (B, 2, H, W)
                Channel 0: horizontal component (u)
                Channel 1: vertical component (v)
            image (Tensor): Reference image (B, C, H, W) for boundary detection

        Returns:
            Scalar smoothness loss
        """
        # Compute image gradients for edge detection
        img_dx, img_dy = gradient(image)

        # Compute flow field gradients
        flow_dx, flow_dy = gradient(flow)

        # Small epsilon to avoid division by zero
        eps = 1e-6

        # Compute gradient magnitudes (L2 norm of spatial gradient)
        # sqrt(dx^2 + dy^2) ≈ sqrt(dx^2 + eps) for numerical stability
        flow_dx_magnitude = torch.sqrt(flow_dx ** 2 + eps)
        flow_dy_magnitude = torch.sqrt(flow_dy ** 2 + eps)

        # Ensure magnitude is non-negative
        flow_dx_magnitude = flow_dx_magnitude.abs()
        flow_dy_magnitude = flow_dy_magnitude.abs()

        if self.boundary_awareness:
            # Compute edge weighting: weight = exp(-α * |∇I|)
            # At strong edges (high |∇I|), weight ≈ 0 → no smoothness penalty
            # In smooth regions (low |∇I|), weight ≈ 1 → full smoothness penalty

            # Average image gradients across channels to get edge strength
            img_dx_mean = torch.mean(torch.abs(img_dx), dim=1, keepdim=True)  # (B, 1, H, W-1)
            img_dy_mean = torch.mean(torch.abs(img_dy), dim=1, keepdim=True)  # (B, 1, H-1, W)

            # Exponential edge weighting
            weights_dx = torch.exp(-img_dx_mean * self.alpha)
            weights_dy = torch.exp(-img_dy_mean * self.alpha)

            # Apply weights to flow gradients
            loss_dx = weights_dx * flow_dx_magnitude / 2.0
            loss_dy = weights_dy * flow_dy_magnitude / 2.0
        else:
            # Uniform smoothness penalty (no edge awareness)
            loss_dx = flow_dx_magnitude / 2.0
            loss_dy = flow_dy_magnitude / 2.0

        # Average horizontal and vertical components, then combine
        # Factor of 2 in denominator provides normalization
        return loss_dx.mean() / 2.0 + loss_dy.mean() / 2.0

    def forward(self, flow_vec, image, box=None):
        """
        Compute smoothness loss for flow field.

        Note: The 'box' parameter is accepted for API consistency but not used.
        Smoothness penalty is applied uniformly across the entire flow field.

        Args:
            flow_vec (Tensor): Flow field (B, 2, H, W) or list of flow tensors
            image (Tensor): Reference image (B, C, H, W)
            box (Tensor or None): Not used; accepted for API compatibility

        Returns:
            Scalar smoothness loss
        """
        # Compute smoothness on provided flow field
        # Note: box parameter is unused but kept in signature for consistency
        # with PhotometricLossWeighted and potential future extensions
        return self.func_smooth(flow_vec, image)