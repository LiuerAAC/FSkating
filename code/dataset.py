import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms

class SpinClipDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        video = EncodedVideo.from_path(sample["video"])
        clip = video.get_clip(
            start_sec=sample["start"],
            end_sec=sample["end"]
        )

        video_tensor = clip["video"]  # (C, T, H, W)
        video_tensor = video_tensor.float() / 255.0
        video_tensor = self.transform(video_tensor)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return video_tensor, label
