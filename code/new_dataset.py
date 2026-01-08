import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
import torch.nn.functional as F
from pathlib import Path

class SpinClipDataset(Dataset):
    """
    X3D 专用视频数据集
    - samples: 从 samples.pt 读取的列表，每个元素 dict 包含:
        'video_id', 'video_path', 'start_time', 'end_time', 'label'
    - clip_len: 模型需要的帧数长度
    """
    def __init__(self, samples, clip_len=32):
        self.samples = samples
        self.clip_len = clip_len

        # X3D 官方均值/方差，shape (C,1,1,1) 用于广播
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(3,1,1,1)
        self.std  = torch.tensor([0.225,0.225,0.225]).view(3,1,1,1)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
    #     video_path = sample['video_path']
        video_path = sample['video_path']

        # ===============================
        # 1. 读取视频 clip
        # ===============================
        video = EncodedVideo.from_path(str(video_path))
        clip_data = video.get_clip(
            start_sec=sample['start_time'],
            end_sec=sample['end_time']
        )['video']  # 实际是 (C, T, H, W)

        # 转 float
        video_tensor = clip_data.float()

        # ===============================
        # 2. 强制 shape 检查（非常重要）
        # ===============================
        assert video_tensor.dim() == 4, f"Unexpected dim: {video_tensor.shape}"
        assert video_tensor.shape[0] == 3, f"Expected C=3, got {video_tensor.shape}"

        # 当前 shape
        # (C, T, H, W)
        C, T, H, W = video_tensor.shape

        # ===============================
        # 3. 帧数插值到 clip_len
        # ===============================
        if T != self.clip_len:
            video_tensor = F.interpolate(
                video_tensor.unsqueeze(0),   # (1, C, T, H, W)
                size=(self.clip_len, H, W),
                mode="trilinear",
                align_corners=False
            ).squeeze(0)                      # (C, clip_len, H, W)

        # 再次确认
        assert video_tensor.shape[0] == 3, f"After T interp: {video_tensor.shape}"
        assert video_tensor.shape[1] == self.clip_len

        # ===============================
        # 4. 空间分辨率 resize（非常关键）
        # X3D 官方默认 224×224
        # ===============================
        video_tensor = F.interpolate(
            video_tensor.unsqueeze(0),       # (1, C, T, H, W)
            size=(self.clip_len, 224, 224),
            mode="trilinear",
            align_corners=False
        ).squeeze(0)                          # (C, T, 224, 224)

        # ===============================
        # 5. 归一化（通道维是 dim=0）
        # ===============================
        mean = self.mean.to(video_tensor.device)
        std  = self.std.to(video_tensor.device)

        video_tensor = (video_tensor - mean) / std

        # ===============================
        # 6. 标签
        # ===============================
        label = torch.tensor(sample["label"], dtype=torch.long)

        return video_tensor, label
