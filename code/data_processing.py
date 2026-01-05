import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms

# pip install pytorchvideo torchvision av
# 用 EncodedVideo + 时间裁剪。

LABEL_MAP = {
    "Jump In": 0,
    "Step In": 0,
    "Difficult In": 0,
    "BC": 1,
    "CF": 1,
    "CS": 1,
    "CU": 1,
    "SF": 2,
    "SS": 2,
    "SB": 2,
    "UF": 3,
    "US": 3,
    "UB": 3,
    "UL": 3,
    "Trans": 4,
    "Step Out": 5,
    "Jump Out": 5,
    "Difficult Out": 5,
    "JCP": 6,
}

dataset_root = Path("Dataset")
video_root = dataset_root / "video"
note_file = dataset_root / "note.txt"

samples = []

def load_samples(dataset_root="Dataset"):
    dataset_root = Path(dataset_root)
    video_root = dataset_root / "video"
    note_file = dataset_root / "note.txt"

    samples = []

    with open(note_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            spin = row["Spin"].strip()
            if spin == "":
                continue

            spin_name = spin.split("[")[0].strip()
            if spin_name not in LABEL_MAP:
                continue

            start = float(row["Begin Time - ss.msec"])
            end = float(row["End Time - ss.msec"])
            video_name = row["File"].replace(".eaf", ".mp4")

            video_path = video_root / video_name
            if not video_path.exists():
                continue

            samples.append({
                "video": str(video_path),
                "start": start,
                "end": end,
                "label": LABEL_MAP[spin_name],
            })

    print(f"[INFO] Total samples: {len(samples)}")
    return samples


if __name__ == "__main__":
    load_samples()
