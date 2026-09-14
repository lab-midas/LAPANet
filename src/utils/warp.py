import torch
import torch.nn.functional as F
from einops import rearrange

def warp_rigid_flow(x, flo):
    """
    Warp an image tensor using a rigid (affine) flow.

    Args:
        x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        flo (torch.Tensor): Flow tensor of shape (B, 2, H, W) representing
                            translation offsets (dx, dy) for each pixel.

    Returns:
        torch.Tensor: Warped image tensor of shape (B, C, H, W).
    """
    # Get batch size, channels, height, width from input tensor
    B, C, H, W = x.size()

    # Define a base affine transformation matrix (identity + translation)
    # Shape: (2, 3) -> [[1, 0, 0], [0, 1, 0]]
    theta_sample = torch.tensor([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.float)

    # Expand the base matrix to match the batch size: (B, 2, 3)
    theta_batch = theta_sample.unsqueeze(0).repeat(B, 1, 1).cuda()

    # Scale the flow values to normalized coordinates.
    # The flow is assumed to be in pixel units; dividing by W scales x-axis,
    # but note: the original code divides both dx and dy by W.
    flo_scaled = 2 * flo / W

    # Replace the translation part of the affine matrix with the scaled flow.
    # theta_batch[:, :, -1] corresponds to the last column (tx, ty).
    theta_batch[:, :, -1] = flo_scaled

    # Define the output size for the affine grid
    size = torch.Size((B, C, H, W))

    # Generate the sampling grid using the affine transformation
    grid = F.affine_grid(theta_batch, size, align_corners=True).cuda()

    # Perform grid sampling to warp the input image
    output = F.grid_sample(x.float(), grid, align_corners=True)

    # Create a mask of ones with the same shape as input to handle out-of-bounds
    mask = torch.ones(x.size(), dtype=x.dtype)
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()

    # Warp the mask using the same grid
    mask = torch.nn.functional.grid_sample(mask.float(), grid, align_corners=True)

    # Binarize the mask: pixels that are not fully inside (value < 0.9999) become 0
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    # Apply the mask to zero out invalid regions
    output = output * mask
    return output


def warp_non_rigid_flow(x, flo):
    """
    Warp an image tensor using a non-rigid (dense) flow field.

    Args:
        x (torch.Tensor): Input image tensor of shape (B, C, H, W) or
                          (B, F, C, H, W) where F is the number of frames.
        flo (torch.Tensor): Flow tensor of shape (B, 2, H, W) or
                            (B, F, 2, H, W) representing per-pixel offsets.

    Returns:
        torch.Tensor: Warped image tensor of shape (B, C, H, W) or
                      (B, F, C, H, W) matching the input dimensionality.
    """
    # If input is 5D (batch, frames, channels, height, width),
    # merge frames and channels for processing, and remember to reshape back.
    if x.dim() == 5:
        f = x.shape[1]
        x = rearrange(x, 'b f c h w -> b (f c) h w')
        reshape = True
    else:
        reshape = False

    # If flow is 5D (batch, frames, 2, height, width),
    # merge batch and frames dimensions for processing.
    if flo.dim() == 5:
        flo = rearrange(flo, 'b f c h w -> (b f) c h w')

    # Get dimensions from the (possibly reshaped) input tensor
    B, C, H, W = x.size()

    # Ensure input is positive and float (absolute value taken)
    x = torch.abs(x).float()

    # Create a mesh grid of pixel coordinates.
    # xx: horizontal coordinates (0 to W-1), yy: vertical coordinates (0 to H-1)
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

    # Concatenate xx and yy to form a grid of shape (B, 2, H, W)
    grid = torch.cat((xx, yy), 1).float()

    # Create a mask of ones with the same shape as input
    mask = torch.ones(x.size(), dtype=x.dtype)
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()

    # Add the flow to the base grid to get the sampling coordinates
    vgrid = grid + flo

    # Scale the grid coordinates to the range [-1, 1] for grid_sample.
    # Note: uses (W-1) and (H-1) for normalization.
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    # Permute to (B, H, W, 2) as required by grid_sample
    vgrid = vgrid.permute(0, 2, 3, 1)

    # Warp the input image using the non-rigid grid
    output = F.grid_sample(x, vgrid, align_corners=True)

    # Warp the mask using the same grid
    mask = F.grid_sample(mask, vgrid, align_corners=True)

    # Binarize the mask: pixels that are not fully inside become 0
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    # Apply the mask to zero out invalid regions
    out = output * mask

    # If the input was 5D, reshape back to (B, F, C, H, W)
    if reshape:
        out = rearrange(out, 'b (f c) h w -> b f c h w', f=f)

    return out