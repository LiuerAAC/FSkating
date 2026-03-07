import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path

# Import functional library
import A_Final_feature_extract as fe

# ==========================================
# 1. Configuration and Paths
# ==========================================
FEAT_FILE_PATH = "1226_feat.npy"
OUTPUT_JSONL = "final_test_data\1226_cascaded_results.jsonl"

# Model Weights
MSTCN_WEIGHTS = "mstcn_plus_plus.pth"
FACT_WEIGHTS = "fact_refinement.pth"

BUFFER_SECONDS = -4.0 

# ==========================================
# 2. Model Definitions
# ==========================================

# Stage 1: MS-TCN++ (Multi-Stage Temporal Convolutional Network)
class MSTCNPlusPlus(nn.Module):
    """Placeholder for MS-TCN++ Architecture"""
    def __init__(self, num_stages=4, num_layers=10, num_f_maps=64, dim=768, num_classes=36):
        super(MSTCNPlusPlus, self).__init__()
        # MS-TCN++ implementation details usually involve Dual-Dilation or iterative refinement
        # Reusing the existing MultiStage architecture as it follows the TCN pattern
        self.stage1 = PredictionGeneration(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([
            PredictionGeneration(num_layers, num_f_maps, num_classes, num_classes)
            for _ in range(num_stages - 1)
        ])

    def forward(self, x):
        outputs = []
        out = self.stage1(x)
        outputs.append(out)
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        return outputs

# Stage 2: FACT (Feature-Augmented Cross-Transformer)
class FACTModel(nn.Module):
    """Placeholder for FACT Architecture (Refinement Stage)"""
    def __init__(self, input_dim=768, num_classes=36):
        super(FACTModel, self).__init__()
        # FACT typically uses Transformer layers to refine local action boundaries
        self.refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        self.fc_out = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, Channels, Length] -> [Batch, Length, Channels]
        x = x.transpose(1, 2)
        feat = self.refiner(x)
        out = self.fc_out(feat)
        return out.transpose(1, 2) # Back to [B, C, L]

# Helper for TCN-style layers
class PredictionGeneration(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(PredictionGeneration, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i),
                nn.ReLU(),
                nn.Conv1d(num_f_maps, num_f_maps, 1),
                nn.Dropout(0.3)
            ) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers: out = out + layer(out)
        return self.conv_out(out)

# ==========================================
# 3. Utilities
# ==========================================

def get_segments(preds, start_time, inv_map):
    """Convert frame-level predictions to temporal segments"""
    if len(preds) == 0: return []
    segments = []
    curr_label_idx = preds[0]
    curr_start_idx = 0

    for i in range(1, len(preds)):
        if preds[i] != curr_label_idx:
            begin_t = (curr_start_idx * fe.STRIDE) / fe.FPS + start_time
            end_t = (i * fe.STRIDE) / fe.FPS + start_time
            segments.append({
                "label": inv_map.get(curr_label_idx, "Background"),
                "begin": round(begin_t, 3),
                "end": round(end_t, 3)
            })
            curr_label_idx = preds[i]
            curr_start_idx = i
    
    # Handle last segment
    segments.append({
        "label": inv_map.get(curr_label_idx, "Background"),
        "begin": round((curr_start_idx * fe.STRIDE) / fe.FPS + start_time, 3),
        "end": round((len(preds) * fe.STRIDE) / fe.FPS + start_time, 3)
    })
    return [s for s in segments if s["label"] != "Background"]

# ==========================================
# 4. Execution Logic
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Action Maps
    ACTION_MAP = {"Background": 0, "UF": 1, "US": 2, "UB": 3, "SF": 5} # Truncated for brevity
    INV_MAP = {v: k for k, v in ACTION_MAP.items()}

    # 1. Load Models
    print("[*] Initializing Models...")
    mstcn = MSTCNPlusPlus(num_classes=36).to(device)
    fact = FACTModel(input_dim=768, num_classes=36).to(device)
    
    # Load weights (assuming files exist)
    # mstcn.load_state_dict(torch.load(MSTCN_WEIGHTS, map_location=device))
    # fact.load_state_dict(torch.load(FACT_WEIGHTS, map_location=device))
    mstcn.eval()
    fact.eval()

    # 2. Load Global Features
    print(f"[*] Loading feature file: {FEAT_FILE_PATH}")
    full_feats = np.load(FEAT_FILE_PATH)
    feat_tensor = torch.from_numpy(full_feats).float().unsqueeze(0).transpose(1, 2).to(device)

    # 3. Stage 1: Global Temporal Proposal (MS-TCN++)
    print("[Stage 1] Running MS-TCN++ for coarse segmentation...")
    with torch.no_grad():
        stage1_outputs = mstcn(feat_tensor)
        # Use the final stage output for coarse boundaries
        coarse_preds = torch.argmax(stage1_outputs[-1], dim=1).squeeze(0).cpu().numpy()
        coarse_segments = get_segments(coarse_preds, 0, INV_MAP)

    # 4. Stage 2: Fine-grained Refinement (FACT)
    print(f"[Stage 2] Refining {len(coarse_segments)} segments with FACT...")
    refined_results = []
    
    for seg in coarse_segments:
        # Convert time back to feature indices
        idx_start = int((seg['begin'] * fe.FPS) / fe.STRIDE)
        idx_end = int((seg['end'] * fe.FPS) / fe.STRIDE)
        
        # Extract sub-feature for FACT refinement
        feat_part = feat_tensor[:, :, idx_start:idx_end]
        if feat_part.shape[-1] < 1: continue

        with torch.no_grad():
            # FACT processes the proposal and outputs refined frame-level labels
            refined_out = fact(feat_part)
            refined_preds = torch.argmax(refined_out, dim=1).squeeze(0).cpu().numpy()
            
            # Sub-segment the proposal
            fine_segments = get_segments(refined_preds, seg['begin'], INV_MAP)
            
            refined_results.append({
                "original_proposal": seg,
                "refined_segments": fine_segments,
                "clip_start": round(max(0, seg['begin'] - abs(BUFFER_SECONDS)), 3),
                "clip_end": round(seg['end'] + abs(BUFFER_SECONDS), 3)
            })

    # 5. Save Results
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in refined_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[SUCCESS] Cascaded inference (MSTCN++ -> FACT) completed.")
    print(f"Results saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
