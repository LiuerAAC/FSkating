import numpy as np
import torch
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from VideoMAE_load import VideoMAEWrapper

# config
CLIP_LEN = 16
STRIDE = 8
FPS = 30
# CKPT_PATH = 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_video_duration(video_path):
    """obtain actual time"""
    with VideoFileClip(str(video_path)) as clip:
        return clip.duration


def extract_full_video_features(video_path):
    """
    function：extract feature
    return: (T, 768)  numpy 
    """
    duration = get_video_duration(video_path)

    print(f"[*]  initializing VideoMAE (Device: {DEVICE})...")
    videomae = VideoMAEWrapper(ckpt_path=CKPT_PATH, device=DEVICE)

    print(f"[*] Processing: {Path(video_path).name} (Time: {duration:.2f}s)")
    feats = videomae.extract_features(
        video_path=str(video_path),
        clip_start=0,
        clip_end=duration,  
        clip_len=CLIP_LEN,
        stride=STRIDE,
        fps=FPS
    )
    return feats
