# import torch
# from torch.utils.data import Dataset
# from pytorchvideo.data.encoded_video import EncodedVideo
# import torch.nn.functional as F
# from pathlib import Path

# class SpinClipDataset(Dataset):
#     def __init__(self, samples, video_root, clip_len=32):
#         self.samples = samples
#         self.video_root = Path(video_root)
#         self.clip_len = clip_len

#         # 官方均值/方差，shape (C,1,1,1) 方便广播
#         self.mean = torch.tensor([0.45, 0.45, 0.45]).view(3,1,1,1)
#         self.std = torch.tensor([0.225, 0.225, 0.225]).view(3,1,1,1)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         video_path = self.video_root / f"{sample['video_id']}.mp4"

#         # 读取视频帧
#         video = EncodedVideo.from_path(str(video_path))
#         video_tensor = video.get_clip(
#             start_sec=sample['start_time'],
#             end_sec=sample['end_time']
#         )['video']  # (T,H,W,C)

#         # 转成 (C,T,H,W)
#         video_tensor = video_tensor.permute(3,0,1,2).float()
#         print("video_tensor.shape before normalize:", video_tensor.shape)

#         # 归一化
#         mean = torch.tensor([0.45,0.45,0.45], dtype=torch.float32).view(3,1,1,1)
#         std  = torch.tensor([0.225,0.225,0.225], dtype=torch.float32).view(3,1,1,1)
#         video_tensor = (video_tensor - mean) / std

#         label = torch.tensor(sample['label'], dtype=torch.long)
#         return video_tensor, label

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
        video_path = sample['video_path']

        # 读取视频帧
        video = EncodedVideo.from_path(str(video_path))
        clip_data = video.get_clip(
            start_sec=sample['start_time'],
            end_sec=sample['end_time']
        )['video']  # (T,H,W,C)

        # 转成 (C,T,H,W)
        video_tensor = clip_data.permute(3,0,1,2).float()  # (3,T,H,W)

        # 插值到固定帧数 clip_len
        if video_tensor.shape[1] != self.clip_len:
            # video_tensor shape: (C,T,H,W)
            video_tensor = F.interpolate(
                video_tensor.unsqueeze(0),  # (1,C,T,H,W)
                size=(self.clip_len, video_tensor.shape[2], video_tensor.shape[3]),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # (C,clip_len,H,W)

        # 归一化
        video_tensor = (video_tensor - self.mean) / self.std

        # 标签
        label = torch.tensor(sample['label'], dtype=torch.long)

        return video_tensor, label
