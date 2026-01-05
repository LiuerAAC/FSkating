import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from pytorchvideo.models.hub import x3d_s

from prepare_annotations import load_samples
from dataset import SpinClipDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. Load samples
    samples = load_samples("Dataset")

    # 2. Dataset & Loader
    dataset = SpinClipDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 3. Load X3D
    model = x3d_s(pretrained=True)

    num_classes = 6
    model.blocks[-1].proj = nn.Linear(
        model.blocks[-1].proj.in_features,
        num_classes
    )

    model = model.to(device)

    # 4. Optimizer & Loss
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 5. Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for video, label in tqdm(loader, desc=f"Epoch {epoch}"):
            video = video.to(device)
            label = label.to(device)

            out = model(video)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)

        avg_loss = total_loss / len(loader)
        acc = total_correct / total_samples

        print(f"[Epoch {epoch}] loss={avg_loss:.4f}, acc={acc:.4f}")

    # 6. Save model
    torch.save(model.state_dict(), "x3d_spin.pth")
    print("[INFO] Model saved to x3d_spin.pth")


if __name__ == "__main__":
    main()
