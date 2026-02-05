# extract_videomae_features.py
import json
import numpy as np
from pathlib import Path
from VideoMAE_load import VideoMAEWrapper  # 确保你的 VideoMAE_load.py 在同级目录下

# ---- 参数设置 ----
# 这些参数需要与后续 MS-TCN++ 的设置匹配
CLIP_LEN = 16
STRIDE = 8
FPS = 30

# ---- 初始化 VideoMAE ----
# 注意：MacBook 务必指定 device="cpu"
videomae = VideoMAEWrapper(
    ckpt_path="/Users/yuxuancao/Desktop/GraduateDesign/code/VideoMAE/checkpoints/videomae_vit_base_patch16_224.pth",
    device="cpu"
)

# ---- 主函数 ----
def main():
    # 路径配置
    meta_path = Path("/Users/yuxuancao/Desktop/GraduateDesign/code/spins_with_buffer1.jsonl")
    feat_dir = Path("/Users/yuxuancao/Desktop/GraduateDesign/code/features")
    
    # 确保输出文件夹存在
    feat_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        print(f"[ERROR] 找不到元数据文件: {meta_path}")
        return

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                spin = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 构造输出文件名：视频ID_spin动作ID.npy
            out_name = f"{spin['video_id']}_spin{spin['spin_id']}.npy"
            out_path = feat_dir / out_name

            # 如果已经提取过，则跳过（方便断点续传）
            if out_path.exists():
                # 可以选择打印跳过信息，但如果文件很多建议精简日志
                # print(f"[SKIP] {out_path} already exists")
                continue

            # ---- 使用 VideoMAEWrapper 提取特征 ----
            # 这里返回的 feats 形状通常是 (T_seq, 768)
            feats = videomae.extract_features(
                video_path=spin["video_path"],
                clip_start=spin["clip_start"],
                clip_end=spin["clip_end"],
                clip_len=CLIP_LEN,
                stride=STRIDE,
                fps=FPS
            )

            # ---- 核心修复：检查 feats 是否为 None ----
            if feats is not None:
                # 只有成功提取到特征（即片段长度足够）才保存
                np.save(out_path, feats)
                print(f"[OK] saved {out_path}, shape={feats.shape}")
            else:
                # 如果返回 None，说明该视频片段长度不足以支撑一个 CLIP_LEN 的窗口
                print(f"[WARN] 片段过短或读取失败，跳过: {out_name}")


if __name__ == "__main__":
    print("--- 特征提取任务开始 ---")
    main()
    print("--- 特征提取任务全部完成 ---")