# import torch
# from torch.utils.data import Dataset
# from pytorchvideo.data.encoded_video import EncodedVideo
# from torchvision import transforms

# class SpinClipDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.Normalize(
#                 mean=[0.45, 0.45, 0.45],
#                 std=[0.225, 0.225, 0.225],
#             )
#         ])

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]

#         video = EncodedVideo.from_path(sample["video"])
#         clip = video.get_clip(
#             start_sec=sample["start"],
#             end_sec=sample["end"]
#         )

#         video_tensor = clip["video"]  # (C, T, H, W)
#         video_tensor = video_tensor.float() / 255.0
#         video_tensor = self.transform(video_tensor)

#         label = torch.tensor(sample["label"], dtype=torch.long)
#         return video_tensor, label

import torch
from torch.utils.data import Dataset
from pathlib import Path
from pytorchvideo.data.encoded_video import EncodedVideo
import torch.nn.functional as F

class SpinClipDataset(Dataset):
    """
    X3D 专用视频数据集
    - samples: 从 samples.pt 读取的列表，每个元素 dict 包含:
        'video_id', 'video_path', 'start_time', 'end_time', 'label'
    - clip_len: 模型需要的帧数长度
    """
    def __init__(self, samples, video_root, clip_len=64):
        self.samples = samples
        self.video_root = Path(video_root)  # ⭐ 必须加
        self.clip_len = clip_len

        # X3D 官方均值/方差，shape (C,1,1,1) 用于广播
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(3,1,1,1)
        self.std  = torch.tensor([0.225,0.225,0.225]).view(3,1,1,1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = self.video_root / f"{sample['video_id']}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        # 读取视频帧
        video = EncodedVideo.from_path(str(video_path))
        clip_data = video.get_clip(
            start_sec=sample['start_time'],
            end_sec=sample['end_time']
        )['video']  # (T,H,W,C)

        # 转成 (C,T,H,W)
        video_tensor = clip_data.permute(3,0,1,2).to(torch.float32)  # (C,T,H,W)

        # 如果帧数不够，插值到 clip_len
        if video_tensor.shape[1] != self.clip_len:
            video_tensor = torch.nn.functional.interpolate(
                video_tensor.unsqueeze(0),  # (1,C,T,H,W)
                size=(self.clip_len, video_tensor.shape[2], video_tensor.shape[3]),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # (C,clip_len,H,W)

        # 归一化 (保证 broadcast)
        mean = torch.tensor([0.45,0.45,0.45], dtype=torch.float32, device=video_tensor.device).view(3,1,1,1)
        std  = torch.tensor([0.225,0.225,0.225], dtype=torch.float32, device=video_tensor.device).view(3,1,1,1)
        video_tensor = (video_tensor - mean) / std

        label = torch.tensor(sample['label'], dtype=torch.long)
        return video_tensor, label
