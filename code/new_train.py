import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from pytorchvideo.models.hub import x3d_s
from dataset import SpinClipDataset  # 上面写好的 SpinClipDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # =====================
    # 1. 加载样本
    # =====================
    samples = torch.load("samples.pt", weights_only=False)

    print(f"[INFO] Loaded {len(samples)} samples")

    # =====================
    # 2. Dataset & DataLoader
    # =====================
    dataset = SpinClipDataset(samples=samples, clip_len=32)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    print(1)
    # =====================
    # 3. 加载 X3D 模型
    # =====================
    model = x3d_s(pretrained=True)
    print(2)

    num_classes = 9  # 根据你的 Spin 分类数量修改
    model.blocks[-1].proj = nn.Linear(
        model.blocks[-1].proj.in_features,
        num_classes
    )
    print(3)
    model = model.to(device)
    print(4)

    # =====================
    # 4. 优化器 & 损失
    # =====================
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # =====================
    # 5. 训练循环
    # =====================
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for video, label in tqdm(loader, desc=f"Epoch {epoch}"):
            # video shape: (B,C,T,H,W)
            video = video.to(device)
            label = label.to(device)

            # 前向
            out = model(video)  # (B,num_classes)

            # 损失
            loss = criterion(out, label)

            # 反向 + 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)

        avg_loss = total_loss / len(loader)
        acc = total_correct / total_samples
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}, acc={acc:.4f}")

    # =====================
    # 6. 保存模型
    # =====================
    torch.save(model.state_dict(), "x3d_spin.pth")
    print("[INFO] Model saved to x3d_spin.pth")

if __name__ == "__main__":
    main()
