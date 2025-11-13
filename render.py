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

import torch
from scene import Scene, OffloadSceneDataset
import os
from tqdm import tqdm
from os import makedirs
from torch.utils.data import DataLoader
import torchvision
import imageio
import numpy as np
from utils.general_utils import (
    safe_state,
    set_args,
    set_log_file,
    set_cur_iter,
)
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from strategies.naive_offload import GaussianModelNaiveOffload, naive_offload_eval_one_cam
from strategies.clm_offload import GaussianModelCLMOffload, clm_offload_eval_one_cam
from strategies.no_offload import GaussianModelNoOffload, baseline_accumGrads_micro_step
from arguments import (
    AuxiliaryParams,
    ModelParams,
    PipelineParams,
    OptimizationParams,
    BenchmarkParams,
    DebugParams,
    print_all_args,
    init_args,
)
import utils.general_utils as utils


class ChunkedVideoWriter:
    """
    Video writer that automatically creates new video chunks when max frames is reached.
    
    Args:
        base_path: Base path for video files (e.g., "output/train/renders")
        base_name: Base name for video files (e.g., "render")
        fps: Frames per second for the video
        max_frames_per_chunk: Maximum number of frames per video chunk
        quality: Video quality (1-10, where 10 is best)
    """
    def __init__(self, base_path, base_name, fps=30, max_frames_per_chunk=1000, quality=8):
        self.base_path = base_path
        self.base_name = base_name
        self.fps = fps
        self.max_frames_per_chunk = max_frames_per_chunk
        self.quality = quality
        
        self.current_chunk = 0
        self.current_frame_count = 0
        self.writer = None
        
        makedirs(base_path, exist_ok=True)
        self._create_new_chunk()
    
    def _create_new_chunk(self):
        """Create a new video chunk."""
        if self.writer is not None:
            self.writer.close()
        
        video_filename = os.path.join(
            self.base_path,
            f"{self.base_name}_chunk{self.current_chunk:04d}.mp4"
        )
        
        # Use imageio with ffmpeg backend for better quality
        self.writer = imageio.get_writer(
            video_filename,
            fps=self.fps,
            codec='libx264',
            quality=self.quality,
            pixelformat='yuv420p',
            macro_block_size=1
        )
        
        self.current_frame_count = 0
        print(f"Creating new video chunk: {video_filename}")
    
    def add_frame(self, frame):
        """
        Add a frame to the video. Automatically creates new chunk if needed.
        
        Args:
            frame: numpy array or torch tensor (H, W, C) in range [0, 1]
        """
        # Convert torch tensor to numpy if needed
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        
        # Ensure frame is in [0, 255] uint8 format
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        
        # Handle channel ordering (C, H, W) -> (H, W, C)
        if frame.shape[0] == 3 or frame.shape[0] == 1:
            frame = np.transpose(frame, (1, 2, 0))
        
        # Convert grayscale to RGB if needed
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        
        self.writer.append_data(frame)
        self.current_frame_count += 1
        
        # Check if we need to start a new chunk
        if self.current_frame_count >= self.max_frames_per_chunk:
            self.current_chunk += 1
            self._create_new_chunk()
    
    def close(self):
        """Close the video writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        
        total_chunks = self.current_chunk + (1 if self.current_frame_count > 0 else 0)
        print(f"Video writing complete: {total_chunks} chunk(s) created")


def render_set(model_path, name, iteration, views_info, gaussians, pipeline, background, scene):
    args = utils.get_args()
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    set_cur_iter(iteration)
    generated_cnt = 0

    num_cameras = len(views_info)
    
    # Initialize video writers if video mode is enabled
    video_writer_render = None
    video_writer_gt = None
    if args.render_video:
        video_base_path = os.path.join(model_path, name, "ours_{}".format(iteration), "videos")
        makedirs(video_base_path, exist_ok=True)
        
        video_writer_render = ChunkedVideoWriter(
            base_path=video_base_path,
            base_name=f"{name}_render",
            fps=args.video_fps,
            max_frames_per_chunk=args.max_frames_per_video,
            quality=args.video_quality
        )
        
        if args.save_gt_video:
            video_writer_gt = ChunkedVideoWriter(
                base_path=video_base_path,
                base_name=f"{name}_gt",
                fps=args.video_fps,
                max_frames_per_chunk=args.max_frames_per_video,
                quality=args.video_quality
            )
        
        print(f"Video mode enabled: {name} set will be saved as video(s)")
        print(f"  FPS: {args.video_fps}, Max frames per chunk: {args.max_frames_per_video}")
    
    # Initialize dataset and dataloader
    eval_dataset = OffloadSceneDataset(views_info)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        collate_fn=(lambda batch: batch)
    )
    dataloader_iter = iter(dataloader)
    
    progress_bar = tqdm(
        range(1, num_cameras + 1),
        desc="Rendering progress",
    )
    
    for idx in range(1, num_cameras + 1, 1):
        progress_bar.update(1)
        
        # Load camera from dataloader
        try:
            batched_cameras = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batched_cameras = next(dataloader_iter)
        
        # Load ground-truth images to GPU
        for camera in batched_cameras:
            camera.original_image = camera.original_image_backup.cuda()
        
        for cam_id, camera in enumerate(batched_cameras):
            actual_idx = idx + cam_id
            
            # Apply filtering based on command-line arguments
            if args.sample_freq != -1 and actual_idx % args.sample_freq != 0:
                continue
            if generated_cnt == args.generate_num:
                break
            if os.path.exists(
                os.path.join(render_path, "{0:05d}".format(actual_idx) + ".png")
            ):
                continue
            if args.l != -1 and args.r != -1:
                if actual_idx < args.l or actual_idx >= args.r:
                    continue
            
            # Prepare camera matrices on GPU
            camera.world_view_transform = camera.world_view_transform.cuda()
            camera.full_proj_transform = camera.full_proj_transform.cuda()
            camera.K = camera.create_k_on_gpu()
            camera.camtoworlds = torch.inverse(camera.world_view_transform.transpose(0, 1)).unsqueeze(0)
            
            # Render using appropriate offload strategy
            if args.naive_offload:
                rendered_image = naive_offload_eval_one_cam(
                    camera=camera,
                    gaussians=gaussians,
                    background=background,
                    scene=scene
                )
            elif args.clm_offload:
                rendered_image = clm_offload_eval_one_cam(
                    camera=camera,
                    gaussians=gaussians,
                    background=background,
                    scene=scene
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
                    mode="eval"
                )
            else:
                raise ValueError("Invalid offload configuration")
            
            generated_cnt += 1
            
            if (
                rendered_image is None or len(rendered_image.shape) == 0
            ):  # The image is not rendered locally.
                rendered_image = torch.zeros(
                    camera.original_image.shape, device="cuda", dtype=torch.float32
                )
            
            image = torch.clamp(rendered_image, 0.0, 1.0)
            gt_image = torch.clamp(camera.original_image / 255.0, 0.0, 1.0)
            
            # Save individual PNG images (unless video-only mode is enabled)
            if not args.video_only:
                torchvision.utils.save_image(
                    image,
                    os.path.join(render_path, "{0:05d}".format(actual_idx) + ".png"),
                )
                torchvision.utils.save_image(
                    gt_image,
                    os.path.join(gts_path, "{0:05d}".format(actual_idx) + ".png"),
                )
            
            # Add frames to video if video mode is enabled
            if video_writer_render is not None:
                video_writer_render.add_frame(image)
            
            if video_writer_gt is not None:
                video_writer_gt.add_frame(gt_image)
            
            camera.original_image = None
        
        if generated_cnt == args.generate_num:
            break
    
    # Close video writers if they were created
    if video_writer_render is not None:
        video_writer_render.close()
    
    if video_writer_gt is not None:
        video_writer_gt.close()


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        args = utils.get_args()
        
        # Select the appropriate Gaussian model based on offload strategy
        if args.naive_offload:
            gaussians = GaussianModelNaiveOffload(sh_degree=dataset.sh_degree, only_for_rendering=True)
            utils.print_rank_0("Using GaussianModelNaiveOffload")
        elif args.clm_offload:
            gaussians = GaussianModelCLMOffload(sh_degree=dataset.sh_degree, only_for_rendering=True)
            utils.print_rank_0("Using GaussianModelCLMOffload")
        elif args.no_offload:
            gaussians = GaussianModelNoOffload(sh_degree=dataset.sh_degree, only_for_rendering=True)
            utils.print_rank_0("Using GaussianModelNoOffload (no offload, GPU-only)")
        else:
            raise ValueError(f"Invalid offload configuration: naive_offload={args.naive_offload}, clm_offload={args.clm_offload}, no_offload={args.no_offload}")
        
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False) # TODO: have loading_plan? 

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCamerasInfo(),
                gaussians,
                pipeline,
                background,
                scene,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCamerasInfo(),
                gaussians,
                pipeline,
                background,
                scene,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    ap = AuxiliaryParams(parser)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--generate_num", default=-1, type=int)
    parser.add_argument("--sample_freq", default=-1, type=int)
    parser.add_argument("--l", default=-1, type=int)
    parser.add_argument("--r", default=-1, type=int)
    
    # Video rendering arguments
    parser.add_argument("--render_video", action="store_true", 
                        help="Enable video rendering mode")
    parser.add_argument("--video_fps", default=30, type=int,
                        help="Frames per second for video (default: 30)")
    parser.add_argument("--max_frames_per_video", default=1000, type=int,
                        help="Maximum frames per video chunk (default: 1000)")
    parser.add_argument("--video_quality", default=8, type=int,
                        help="Video quality 1-10, where 10 is best (default: 8)")
    parser.add_argument("--save_gt_video", action="store_true",
                        help="Also save ground truth images as video (only with --render_video)")
    parser.add_argument("--video_only", action="store_true",
                        help="Only save videos, skip saving individual PNG files (only with --render_video)")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    # Validate video arguments
    if args.video_only and not args.render_video:
        print("Warning: --video_only requires --render_video. Enabling video rendering.")
        args.render_video = True
    
    if args.save_gt_video and not args.render_video:
        print("Warning: --save_gt_video requires --render_video. Ignoring --save_gt_video.")
        args.save_gt_video = False
    
    if args.render_video:
        print(f"\n{'='*60}")
        print("Video Rendering Mode Enabled")
        print(f"{'='*60}")
        print(f"  FPS: {args.video_fps}")
        print(f"  Max frames per chunk: {args.max_frames_per_video}")
        print(f"  Video quality: {args.video_quality}/10")
        print(f"  Save ground truth video: {args.save_gt_video}")
        print(f"  Video only (skip PNGs): {args.video_only}")
        print(f"{'='*60}\n")

    # Create log folder
    os.makedirs(args.model_path, exist_ok=True)
    log_file = open(
        args.model_path + "/render.log",
        "w",
    )
    set_log_file(log_file)

    ## Prepare arguments.
    # Check arguments
    init_args(args)
    if args.skip_train:
        args.num_train_cameras = 0
    if args.skip_test:
        args.num_test_cameras = 0
    # Set up global args
    set_args(args)

    print_all_args(args, log_file)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        lp.extract(args),
        args.iteration,
        pp.extract(args),
        args.skip_train,
        args.skip_test,
    )
    
    log_file.close()
    print("\nRendering complete.")
