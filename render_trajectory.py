#!/usr/bin/env python3
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""
CPU-Offloaded Trajectory Rendering for 3D Gaussian Splatting

This script performs trajectory rendering with CPU offloading using trained 3D Gaussian Splatting
models saved as PLY files. It enables rendering of large-scale scenes that may not fit entirely
in GPU memory by keeping Gaussian parameters in CPU RAM and loading them to GPU on-demand.

=== OVERVIEW ===
This script loads a trained 3D Gaussian Splatting model (saved as PLY point cloud files during
training) and generates smooth camera trajectory videos for visualization. It's designed for
scenes with millions or billions of Gaussians where GPU memory is a constraint.

=== KEY FEATURES ===
- Single GPU rendering (no distributed training/inference setup required)
- Three offloading strategies:
  * clm_offload: Chunk-based loading with optimized memory management (recommended for large scenes)
  * naive_offload: Simple CPU↔GPU transfers (baseline offload strategy)
  * no_offload: Keep all data on GPU (fastest but highest memory usage)
- On-demand GPU loading: Transfers data to GPU only during active rendering
- Multiple trajectory types: interpolated, elliptical, spiral, or original training poses
- Video output: Generates MP4 videos at 30 FPS

=== MODEL LOADING ===
IMPORTANT: This script loads models from PLY files (saved during training with scene.save()),
NOT from checkpoint files (.pth). Checkpoint files contain optimizer states and other training
data that are not needed for rendering.

Model files are expected at: {model_path}/point_cloud/iteration_{N}/point_cloud.ply

During training, models are saved using:
- scene.save(iteration) → saves as PLY files (standard format, used by this script)
- scene.save_tensors(iteration) → saves as .pt files (optional fast loading format)

=== USAGE EXAMPLE ===
# Render with CLM offloading (recommended for large scenes):
python render_trajectory.py \\
    --model_path /path/to/model \\
    --source_path /path/to/dataset \\
    --iteration 30000 \\
    --traj_path ellipse \\
    --n_frames 240 \\
    --clm_offload

# Render with no offloading (for scenes that fit in GPU memory):
# Use --iteration -1 to automatically load the latest saved iteration
python render_trajectory.py \\
    --model_path /path/to/model \\
    --source_path /path/to/dataset \\
    --iteration -1 \\
    --traj_path spiral \\
    --n_frames 120 \\
    --no_offload

=== TRAJECTORY TYPES ===
- "interp": Smooth interpolation between training camera poses (good for general use)
- "ellipse": Elliptical orbit around scene center at constant height (good for object-centric scenes)
- "spiral": 3D spiral path through scene bounds (good for scene exploration)
- "original": Use original training camera poses (good for validation/comparison)

=== TECHNICAL DETAILS ===
3D Gaussian Splatting represents scenes as collections of 3D Gaussian primitives. Each Gaussian has:
- Position: 3D coordinates (xyz) - 3 floats
- Rotation: Quaternion - 4 floats
- Scale: 3D anisotropic scaling - 3 floats
- Opacity: Transparency - 1 float
- Color: Spherical harmonics coefficients - (K coefficients × 3 RGB channels) floats

For large scenes (billions of Gaussians), storing all parameters on GPU exceeds available VRAM.
This script's offloading strategies enable rendering by:
1. Storing parameters in CPU RAM
2. Loading required data to GPU per frame (or per chunk)
3. Rendering the frame
4. Clearing GPU memory and moving to next frame

This allows rendering of arbitrarily large scenes on consumer GPUs.
"""

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Optional, List
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    init_args,
)

from scene import Scene, load_scene_info_for_rendering
from scene.cameras import Camera
from strategies.naive_offload import GaussianModelNaiveOffload, naive_offload_eval_one_cam
from strategies.clm_offload import GaussianModelCLMOffload, clm_offload_eval_one_cam
from strategies.no_offload import GaussianModelNoOffload, baseline_accumGrads_micro_step

from traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)

from utils.general_utils import safe_state
import utils.general_utils as utils
from utils.graphics_utils import getWorld2View2
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import loadCam
import psutil


def create_camera_from_c2w(
    c2w: torch.Tensor,
    FoVx: float,
    FoVy: float,
    width: int,
    height: int,
    uid: int = 0,
    image_name: str = "trajectory_frame",
):
    """
    Create a Camera object from camera-to-world transform.
    
    Args:
        c2w: Camera-to-world transformation matrix [4, 4]
        FoVx: Horizontal field of view in radians
        FoVy: Vertical field of view in radians
        width: Image width
        height: Image height
        uid: Camera unique identifier
        image_name: Name for this camera/frame
    
    Returns:
        Camera object
    """
    # Convert c2w to R and T for Camera class
    c2w_np = c2w.cpu().numpy() if isinstance(c2w, torch.Tensor) else c2w
    
    # # Extract rotation matrix (first 3x3) and translation (last column)
    # R = c2w_np[:3, :3].T  # Camera uses world-to-camera rotation
    # T = -R @ c2w_np[:3, 3]  # Camera uses world-to-camera translation

    R = c2w_np[:3, :3]
    T = c2w_np[:3, 3]
    
    # Create dummy image (not needed for rendering)
    dummy_image = torch.zeros((3, height, width), dtype=torch.float32)
    
    camera = Camera(
        colmap_id=uid,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=image_name,
        uid=uid,
        offload=True,  # Keep camera on CPU
    )
    Camera.original_image_backup = None # delete the image

    # if the system cpu memory is low, delete the camera
    if psutil.virtual_memory().percent > 90: # 90%
        print("System CPU memory is low, exiting...")
        exit()
    return camera


def generate_trajectory_cameras(
    world_to_camera_all: list,
    camtoworlds_all: list,
    traj_path: str = "interp",
    n_frames: int = 120,
    FoVx: Optional[float] = None,
    FoVy: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """
    Generate camera trajectory for rendering.
    
    Args:
        world_to_camera_all: List of world-to-camera transformation matrices
        camtoworlds_all: List of camera-to-world transformation matrices
        traj_path: Type of trajectory ("interp", "ellipse", "spiral", "original")
        n_frames: Number of frames to generate (ignored for "original")
        FoVx, FoVy: Field of view (uses first training camera if None)
        width, height: Image size (uses first training camera if None)
    
    Returns:
        List of Camera objects for trajectory
    """
    # Check if we have training cameras
    if len(world_to_camera_all) == 0:
        raise ValueError("No training cameras found in scene")
    
    # stack camtoworlds_all to get train_c2w
    train_c2w = np.stack(camtoworlds_all, axis=0)
    # Generate trajectory based on type
    if traj_path == "original":
        print(f"Using original training poses: {len(train_c2w)} cameras")
        c2w_traj = train_c2w[:n_frames, :3, :] # only keep the first n_frames poses
    else:
        # Use subset of poses for trajectory generation (skip first/last few)
        train_c2w_subset = train_c2w[5:-5] if len(train_c2w) > 10 else train_c2w
        
        if traj_path == "interp":
            c2w_traj = generate_interpolated_path(
                train_c2w_subset[:, :3, :], n_interp=n_frames // len(train_c2w_subset)
            )
        elif traj_path == "ellipse":
            height_z = train_c2w_subset[:, 2, 3].mean()
            c2w_traj = generate_ellipse_path_z(
                train_c2w_subset[:, :3, :], n_frames=n_frames, height=height_z
            )
        elif traj_path == "spiral":
            # Compute bounds from training camera positions
            positions = train_c2w_subset[:, :3, 3]
            bounds = np.array([
                positions.min(axis=0) - 1.0,
                positions.max(axis=0) + 1.0
            ])
            c2w_traj = generate_spiral_path(
                train_c2w_subset[:, :3, :], bounds=bounds, n_frames=n_frames
            )
        else:
            raise ValueError(f"Unsupported trajectory type: {traj_path}")
    
    # Convert trajectory to homogeneous coordinates [N, 4, 4]
    c2w_traj_homo = np.concatenate([
        c2w_traj,
        np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(c2w_traj), axis=0),
    ], axis=1)
    
    # Create Camera objects for each pose
    # import pdb; pdb.set_trace()
    trajectory_cameras = []
    for idx, c2w in enumerate(c2w_traj_homo):
        camera = create_camera_from_c2w(
            c2w=torch.from_numpy(c2w).float(),
            FoVx=FoVx,
            FoVy=FoVy,
            width=width,
            height=height,
            uid=idx,
            image_name=f"traj_frame_{idx:05d}",
        )
        trajectory_cameras.append(camera)
    
    print(f"Generated trajectory with {len(trajectory_cameras)} cameras")
    return trajectory_cameras


def render_trajectory_video(
    args,
    scene: Scene,
    gaussians,
    pipe_args,
    trajectory_cameras: List[Camera],
    output_path: str,
    background: torch.Tensor,
    render_depth: bool = True,
):
    """
    Render trajectory and save as video.
    
    Args:
        args: Command-line arguments
        scene: Scene object
        gaussians: Gaussian model
        pipe_args: Pipeline parameters
        trajectory_cameras: List of cameras for trajectory
        output_path: Path to save output video
        background: Background color tensor
        render_depth: Whether to include depth visualization
    """
    # Limit number of frames if specified
    total_frames = len(trajectory_cameras)
    # Create video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=30)
    
    print(f"Rendering {total_frames} frames...")
    
    # Render each frame
    for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
        camera = trajectory_cameras[frame_idx]
        
        # Prepare camera matrices on GPU
        camera.world_view_transform = camera.world_view_transform.cuda()
        camera.full_proj_transform = camera.full_proj_transform.cuda()
        camera.K = camera.create_k_on_gpu()
        camera.camtoworlds = torch.inverse(
            camera.world_view_transform.transpose(0, 1)
        ).unsqueeze(0)
        
        # import pdb; pdb.set_trace()
        # Render frame based on offload strategy
        if args.naive_offload:
            rendered_image = naive_offload_eval_one_cam(
                gaussians=gaussians,
                scene=scene,
                camera=camera,
                background=background,
            )
        elif args.clm_offload:
            rendered_image = clm_offload_eval_one_cam(
                camera=camera,
                gaussians=gaussians,
                background=background,
                scene=scene,
            )
        elif args.no_offload:
            rendered_image, _, _, _ = baseline_accumGrads_micro_step(
                means3D=gaussians.get_xyz,
                opacities=gaussians.get_opacity,
                scales=gaussians.get_scaling,
                rotations=gaussians.get_rotation,
                shs=gaussians.get_features,
                sh_degree=gaussians.active_sh_degree,
                camera=camera,
                background=background,
                mode="eval",
            )
        else:
            raise ValueError("Invalid offload configuration")
        
        # Process rendered output
        colors = torch.clamp(rendered_image, 0.0, 1.0)  # [H, W, 3] or [3, H, W]
        
        # Ensure colors are in [H, W, 3] format
        if colors.shape[0] == 3:
            colors = colors.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        
        # Convert to numpy for video writing
        colors_np = (colors * 255).cpu().to(torch.uint8).numpy()
        
        # Write frame to video
        writer.append_data(colors_np)
        
        # Clean up GPU memory
        del rendered_image, colors, colors_np
        camera.world_view_transform = camera.world_view_transform.cpu()
        camera.full_proj_transform = camera.full_proj_transform.cpu()
        torch.cuda.empty_cache()
    
    writer.close()
    print(f"Saved video: {output_path}")


def main():
    """Main function for trajectory rendering."""
    # Parse command-line arguments
    parser = ArgumentParser(description="Trajectory rendering script")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    
    # Add trajectory-specific arguments
    parser.add_argument("--traj_path", type=str, default="original",
                       choices=["interp", "ellipse", "spiral", "original"],
                       help="Type of camera trajectory")
    parser.add_argument("--n_frames", type=int, default=120,
                       help="Number of frames to generate (ignored for 'original')")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: model_path/render_traj)")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="Iteration to load (default: -1 = load latest saved iteration)")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Initialize arguments
    init_args(args)
    args = utils.get_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "render_traj")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = open(os.path.join(args.output_dir, "render_traj.log"), "w")
    utils.set_log_file(log_file)
    
    utils.print_rank_0("=" * 80)
    utils.print_rank_0("Trajectory Rendering")
    utils.print_rank_0("=" * 80)
    utils.print_rank_0(f"Model path: {args.model_path}")
    utils.print_rank_0(f"Load iteration: {args.iteration} (-1 = latest)")
    utils.print_rank_0(f"Trajectory type: {args.traj_path}")
    utils.print_rank_0(f"Number of frames: {args.n_frames}")
    utils.print_rank_0(f"Output directory: {args.output_dir}")
    utils.print_rank_0("=" * 80)
    
    # Initialize Gaussian model based on offload strategy
    dataset_args = lp.extract(args)
    
    if args.naive_offload:
        gaussians = GaussianModelNaiveOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelNaiveOffload")
    elif args.clm_offload:
        gaussians = GaussianModelCLMOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelCLMOffload")
    elif args.no_offload:
        gaussians = GaussianModelNoOffload(sh_degree=dataset_args.sh_degree)
        utils.print_rank_0("Using GaussianModelNoOffload (no offload, GPU-only)")
    else:
        raise ValueError(
            f"Invalid offload configuration: naive_offload={args.naive_offload}, "
            f"clm_offload={args.clm_offload}, no_offload={args.no_offload}"
        )
    
    # Load scene info and trained model
    # Note: We load camera metadata WITHOUT decoding images to disk (rendering-only workflow)
    # The PLY files are saved at: model_path/point_cloud/iteration_{iteration}/point_cloud.ply
    with torch.no_grad():
        utils.print_rank_0(f"\nLoading scene info and model from iteration {args.iteration}...")
        
        # Load scene information (camera poses, intrinsics) without image decoding
        scene_info, cameras_extent = load_scene_info_for_rendering(args)
        utils.print_rank_0("Loaded scene metadata (camera poses and intrinsics)")
        
        if args.load_pt_path != '':
            gaussians.load_tensors(args.load_pt_path)
            utils.print_rank_0(f"Loaded Gaussians from {args.load_pt_path}")
        elif args.load_ply_path != '':
            gaussians.load_ply(args.load_ply_path)
            utils.print_rank_0(f"Loaded Gaussians from {args.load_ply_path}")
        else:
            if args.iteration == -1:
                loaded_iter = searchForMaxIteration(
                    os.path.join(args.model_path, "point_cloud")
                )
            else:
                loaded_iter = args.iteration
            ply_path = os.path.join(
                args.model_path, "point_cloud", f"iteration_{loaded_iter}"
            )
            gaussians.load_ply(ply_path)
            utils.print_rank_0(f"Loaded Gaussians from iteration {loaded_iter}...") 

        utils.print_rank_0(f"Loaded {len(gaussians._xyz)} Gaussians.")
        
        # Set active SH degree to maximum for best quality rendering
        # During training, SH degree is progressively increased
        # For rendering, we use the maximum degree for best visual quality
        gaussians.active_sh_degree = gaussians.max_sh_degree
    
    # import pdb; pdb.set_trace()

    # know the image width and height
    image_width = scene_info.train_cameras[0].width
    image_height = scene_info.train_cameras[0].height

    # utils.get_img_width()
    utils.set_img_size(image_height, image_width)

    # Setup background
    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None

    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    world_to_camera_all = []
    camtoworlds_all = []
    # K_all = []
    def getWorld2View(R, t):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        # cam_center = C2W[:3, 3]
        # cam_center = (cam_center + translate) * scale
        # C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    # def getK(FovX, FovY, image_width, image_height):
    #     tanfovx = math.tan(FovX * 0.5)
    #     tanfovy = math.tan(FovY * 0.5)
    #     focal_length_x = image_width / (2 * tanfovx)
    #     focal_length_y = image_height / (2 * tanfovy)
    #     K = torch.tensor(
    #         [
    #             [focal_length_x, 0, image_width / 2.0],
    #             [0, focal_length_y, image_height / 2.0],
    #             [0, 0, 1],
    #         ],
    #         # device="cuda",
    #     )
    #     return K

    for id, cam_info in tqdm(
        enumerate(scene_info.train_cameras), total=len(scene_info.train_cameras)
    ):
        world_view_transform = (
            torch.tensor(getWorld2View(cam_info.R, cam_info.T)).transpose(0, 1)
        )
        world_to_camera_all.append(world_view_transform)
        view_world_transform = world_view_transform.t().inverse()
        camtoworlds_all.append(view_world_transform)
        # K_all.append(getK(cam_info.FovX, cam_info.FovY, image_width, image_height))
    FoVx = scene_info.train_cameras[0].FovX
    FoVy = scene_info.train_cameras[0].FovY

    # import pdb; pdb.set_trace()
    # Generate trajectory cameras
    trajectory_cameras = generate_trajectory_cameras(
        world_to_camera_all=world_to_camera_all,
        camtoworlds_all=camtoworlds_all,
        traj_path=args.traj_path,
        n_frames=args.n_frames,
        FoVx=FoVx,
        FoVy=FoVy,
        width=image_width,
        height=image_height,
    )
    
    # Render trajectory video
    pipe_args = pp.extract(args)
    video_path = os.path.join(args.output_dir, f"trajectory_{args.traj_path}.mp4")
    
    render_trajectory_video(
        args=args,
        scene=None,  # No scene object needed for rendering
        gaussians=gaussians,
        pipe_args=pipe_args,
        trajectory_cameras=trajectory_cameras,
        output_path=video_path,
        background=background,
    )
    
    # Cleanup
    log_file.write(f"Rendering completed. Video saved to: {video_path}\n")
    log_file.close()
    utils.print_rank_0("\nRendering complete!")
    utils.print_rank_0(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()
