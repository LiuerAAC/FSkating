import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---- Video读取 ----
import decord
decord.bridge.set_bridge("torch") 

def load_video_segment(video_path, start_sec, end_sec, fps=30):
    """
    更鲁棒的视频读取
    """
    try:
        vr = decord.VideoReader(str(video_path))
        # 获取视频真实的 FPS，而不是手动指定
        real_fps = vr.get_avg_fps()
        total_frames = len(vr)

        # 使用真实 FPS 计算帧索引
        start_f = max(0, int(start_sec * real_fps))
        end_f = min(total_frames - 1, int(end_sec * real_fps))
        
        if start_f >= end_f:
            print(f"  [DEBUG] 索引错误: start_f({start_f}) >= end_f({end_f})")
            return None
            
        # 核心修复：确保请求的索引序列在合法范围内
        frame_indices = list(range(start_f, end_f))
        
        # 如果索引还是超了，做最后的截断
        frame_indices = [idx for idx in frame_indices if idx < total_frames]
        
        if not frame_indices:
            return None

        frames = vr.get_batch(frame_indices)
        return frames.float() / 255.0
        
    except Exception as e:
        # 打印具体的报错原因，看看是不是 decord 崩溃了
        print(f"  [DEBUG] Decord 读取失败: {video_path}, 错误: {e}")
        return None
        
# ---- VideoMAE 模型封装 ----
from VideoMAE.modeling_pretrain import PretrainVisionTransformer

class VideoMAEWrapper:
    def __init__(self, ckpt_path: str, device="cpu"):
        """
        device: 在 Mac 上建议保持 "cpu"，若有 M1/M2/M3 芯片可尝试 "mps"
        """
        self.device = device

        # 1. 定义模型结构 (与 vit_base_patch16 保持一致)
        self.model = PretrainVisionTransformer(
            img_size=224, patch_size=16, encoder_embed_dim=768, encoder_depth=12,
            encoder_num_heads=12, decoder_embed_dim=384, decoder_depth=4,
            decoder_num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=torch.nn.LayerNorm, tubelet_size=2,
        )

        print(f"[INFO] 正在从路径加载权重: {ckpt_path}")
        # 2. 加载权重
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # 自动获取权重字典内容
        if "module" in checkpoint:
            raw_dict = checkpoint["module"]
        elif "model" in checkpoint:
            raw_dict = checkpoint["model"]
        else:
            raw_dict = checkpoint

        # 3. 修正键名映射 (将扁平化权重映射到 encoder 内部)
        new_state_dict = {}
        for k, v in raw_dict.items():
            if not k.startswith("encoder.") and not k.startswith("decoder."):
                # 只有当这些键不是以 encoder/decoder 开头时，才添加前缀
                new_state_dict[f"encoder.{k}"] = v
            else:
                new_state_dict[k] = v

        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"[INFO] 权重对齐结果: {msg}")
        
        self.model.to(self.device)
        self.model.eval()

    def forward_clip(self, clip: torch.Tensor):
        """处理 16 帧的视频片段"""
        if clip.ndim == 4:
            clip = clip.unsqueeze(0) # [B, T, H, W, C]

        # 维度变换: [B, T, H, W, C] -> [B, C, T, H, W]
        # 注意：VideoMAE 官方实现通常期望 [B, C, T, H, W]
        clip = clip.permute(0, 4, 1, 2, 3)
        
        # 尺寸检查与缩放
        B, C, T, H, W = clip.shape
        if H != 224 or W != 224:
            clip = torch.nn.functional.interpolate(
                clip.reshape(B*T, C, H, W), size=(224, 224), mode='bilinear'
            ).reshape(B, C, T, 224, 224)

        clip = clip.to(self.device)

        with torch.no_grad():
            # --- 核心：构造全零 Mask 解决 TypeError ---
            # Patch 数量 = (T//tubelet) * (H//patch) * (W//patch) = 8 * 14 * 14 = 1568
            num_patches = (T // 2) * (224 // 16) * (224 // 16)
            mask = torch.zeros((B, num_patches), device=self.device).bool()

            # 推理过程
            latent = self.model.encoder(clip, mask) 
            # latent shape: [B, 1568, 768] (因为没被遮挡，所有 patch 都在)
            feat = latent.mean(dim=1) # 空间+时间平均池化
            
        return feat.squeeze(0).cpu().numpy()

    def extract_features(self, video_path: str, clip_start: float, clip_end: float, clip_len=16, stride=8, fps=30):
        """带进度条的特征提取"""
        frames = load_video_segment(video_path, clip_start, clip_end, fps=fps)
        if frames is None:
            return None

        total_frames = frames.shape[0]
        if total_frames < clip_len:
            return None

        feats = []
        # 计算滑动窗口起始位置
        indices = list(range(0, total_frames - clip_len + 1, stride))
        
        # 提取过程显示进度条
        pbar = tqdm(indices, desc=f"  -> Processing {Path(video_path).stem}", leave=False)
        for i in pbar:
            clip = frames[i : i + clip_len]
            feat = self.forward_clip(clip)
            feats.append(feat)
            
        return np.stack(feats) if feats else None