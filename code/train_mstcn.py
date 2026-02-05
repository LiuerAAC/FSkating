import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path

# ==========================================
# 1. 核心配置与动作映射
# ==========================================
ACTION_MAP = {
    "Background": 0,
    "UF": 1, "US": 2, "UB": 3, "UL": 4, "SF": 5, "SS": 6, "SB": 7, "CF": 8, "CS": 9, "CU": 10, 
    "Windmill": 11, "Other-NBP": 12, "BU": 13, "BS": 14, "BC": 15, 
    "Jump In": 16, "Step In": 17, "Difficult In": 18, "Jump Out": 19, "Step Out": 20, "Difficult Out": 21,
    "Trans": 22, "CP": 23, "JCP": 24,
    "UF+Inside": 25, "SF+Inside": 26, "CF+Inside": 27, "CS+Inside": 28, "CU+Inside": 29, "BC+Inside": 30,
}

JSONL_PATH = Path("GraduateDesign/code/spins_with_buffer_clean.jsonl")
FEAT_DIR = Path("GraduateDesign/code/features")

FEAT_DIM = 768        
NUM_CLASSES = 30      
BATCH_SIZE = 1        
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
STRIDE = 16           
ORIGINAL_FPS = 30

# ==========================================
# 2. MS-TCN 模型定义 (Single Stage)
# ==========================================
class SingleStageTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i),
                nn.ReLU(),
                nn.Conv1d(num_f_maps, num_f_maps, 1),
                nn.Dropout(0.3)
            ) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = out + layer(out)
        return self.conv_out(out)

# ==========================================
# 3. 数据集加载 (针对 0_spin0.npy 格式)
# ==========================================
class SkatingDataset(Dataset):
    def __init__(self, jsonl_path, feat_dir, action_map):
        self.samples = []
        self.feat_dir = feat_dir
        self.action_map = action_map
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        print(f"成功加载数据集，共 {len(self.samples)} 条旋转序列。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 确保 video_id 是字符串（处理 0, 1, 2...）
        video_id = str(sample['video_id'])
        spin_id = str(sample['spin_id'])
        
        # 匹配文件名格式: 0_spin0.npy
        feat_path = self.feat_dir / f"{video_id}_spin{spin_id}.npy"
        
        if not feat_path.exists():
            # 这里的异常会被 DataLoader 捕获或直接抛出，我们在 train 里处理
            raise FileNotFoundError(f"Missing file: {feat_path}")
            
        feat_np = np.load(feat_path)
        features = torch.from_numpy(feat_np).float()
        
        if features.dim() == 3:
            features = features.squeeze(0)
            
        T = features.shape[0]
        clip_start = sample['clip_start']
        target = np.zeros(T, dtype=np.int64)
        
        effective_fps = ORIGINAL_FPS / STRIDE
        
        for seg in sample['segments']:
            s_idx = int((seg['begin'] - clip_start) * effective_fps)
            e_idx = int((seg['end'] - clip_start) * effective_fps)
            
            label_id = self.action_map.get(seg['label'], 0)
            target[max(0, s_idx):min(T, e_idx)] = label_id
            
        return features.transpose(0, 1), torch.tensor(target)

# ==========================================
# 4. 训练主循环
# ==========================================
def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    dataset = SkatingDataset(JSONL_PATH, FEAT_DIR, ACTION_MAP)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SingleStageTCN(num_layers=10, num_f_maps=64, dim=FEAT_DIM, num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    print("开始训练...")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        processed_batches = 0
        
        for feats, labels in dataloader:
            try:
                feats, labels = feats.to(device), labels.to(device)
                
                optimizer.zero_grad()
                output = model(feats)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                processed_batches += 1
            except FileNotFoundError as e:
                # 打印缺失的文件，但不停止训练
                continue

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / processed_batches if processed_batches > 0 else 0
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - 平均损失: {avg_loss:.4f}")

    torch.save(model.state_dict(), "skating_mstcn_model.pth")
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    train()