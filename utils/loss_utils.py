#
# Copyright (C) 2023, Inria GRAPHDECO research group
# Copyright (C) 2025, New York University
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE file.
#
# Original 3D Gaussian Splatting code from:
# https://github.com/graphdeco-inria/gaussian-splatting
#
# CLM-GS modifications by NYU Systems Group
# https://github.com/nyu-systems/CLM-GS
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def pixelwise_l1_with_mask(img1, img2, pixel_mask):
    # img1, img2: (3, H, W)
    # pixel_mask: (H, W) bool torch tensor as mask.
    # only compute l1 loss for the pixels that are touched

    pixelwise_l1_loss = torch.abs((img1 - img2)) * pixel_mask.unsqueeze(0)
    return pixelwise_l1_loss


def pixelwise_ssim_with_mask(img1, img2, pixel_mask):
    window_size = 11

    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    pixelwise_ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    pixelwise_ssim_loss = pixelwise_ssim_loss * pixel_mask.unsqueeze(0)

    return pixelwise_ssim_loss


def depth_loss(rendered_depth, gt_depth, valid_mask=None):
    """
    Compute depth loss between rendered depth and ground truth depth.

    Args:
        rendered_depth: Rendered depth map (H, W) in camera space (z-depth)
        gt_depth: Ground truth depth map (H, W) in meters
        valid_mask: Optional mask (H, W) where 1=valid, 0=invalid.
                   If None, uses gt_depth > 0 as valid mask.

    Returns:
        Scalar depth loss value
    """
    # Create valid mask: only supervise where gt_depth is valid (> 0)
    if valid_mask is None:
        valid_mask = (gt_depth > 0).float()
    else:
        # Combine with depth validity
        valid_mask = valid_mask * (gt_depth > 0).float()

    # Also mask out where rendered depth is invalid
    valid_mask = valid_mask * (rendered_depth > 0).float()

    # Count valid pixels
    num_valid = valid_mask.sum()
    if num_valid < 1:
        return torch.tensor(0.0, device=rendered_depth.device)

    # L1 depth loss on valid pixels
    depth_diff = torch.abs(rendered_depth - gt_depth) * valid_mask
    loss = depth_diff.sum() / num_valid

    return loss


def depth_loss_log(rendered_depth, gt_depth, valid_mask=None):
    """
    Compute depth loss in log space, which is more robust to scale differences.

    Args:
        rendered_depth: Rendered depth map (H, W) in camera space (z-depth)
        gt_depth: Ground truth depth map (H, W) in meters
        valid_mask: Optional mask (H, W) where 1=valid, 0=invalid.

    Returns:
        Scalar depth loss value in log space
    """
    # Create valid mask: only supervise where gt_depth is valid (> 0)
    if valid_mask is None:
        valid_mask = (gt_depth > 0).float()
    else:
        valid_mask = valid_mask * (gt_depth > 0).float()

    # Also mask out where rendered depth is invalid
    valid_mask = valid_mask * (rendered_depth > 0).float()

    # Count valid pixels
    num_valid = valid_mask.sum()
    if num_valid < 1:
        return torch.tensor(0.0, device=rendered_depth.device)

    # Log-space L1 depth loss
    eps = 1e-6
    log_rendered = torch.log(rendered_depth + eps)
    log_gt = torch.log(gt_depth + eps)
    depth_diff = torch.abs(log_rendered - log_gt) * valid_mask
    loss = depth_diff.sum() / num_valid

    return loss
