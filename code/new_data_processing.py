from pathlib import Path
import torch

# =========================
# 配置
# =========================
ANNOTATION_TXT = "0to34.txt"  # TXT 文件路径
VIDEO_ROOT = Path("Dataset/video")  # 视频文件夹路径
OUTPUT_SAMPLES = "samples.pt"       # 保存的样本列表

# =========================
# Spin -> label 映射
# =========================
spin_label_map = {
    # Camel Spin
    "CF": 0, "CS": 0, "CU": 0,
    # Sit Spin
    "SF": 1, "SS": 1, "SB": 1,
    # Upright Spin
    "UF": 2, "US": 2, "UB": 2, "UL": 2,
    # NBP
    "Windmill": 3, "Other NBP": 4,
    # In
    "Jump In": 5, "Step In":5, "Difficult In": 5,
    # Out
    "Jump Out": 6, "Step Out": 6, "Difficult Out": 6,
    # Change Position
    "JCP": 7, "CP": 7,
    # Trans（可选）
    "Trans": 8
}

# =========================
# 解析 TXT -> samples
# =========================
def parse_txt_to_samples(txt_path, video_root):
    samples = []

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 清理表头空格和 BOM
    header = [h.strip().replace("\ufeff", "") for h in lines[0].strip().split("\t")]
    print(f"Header columns: {header}")

    for line in lines[1:]:
        cols = line.strip().split("\t")
        if len(cols) != len(header):
            continue  # 跳过列数不对的行
        row = dict(zip(header, [c.strip() for c in cols]))

        # 视频文件路径
        video_file = row.get("File", "")
        if not video_file:
            continue
        video_id = int(Path(video_file).stem)
        video_path = video_root / f"{video_id}.mp4"
        if not video_path.exists():
            # print(f"Warning: video {video_path} not found. Skipping.")
            continue

        # Spin -> label
        spin = row.get("Spin", "")
        if not spin:
            continue
        # 处理带 [] 的情况，如 "Jump In [a1]" -> "Jump In"
        spin_clean = spin.split()[0] if "[" in spin else spin
        label = spin_label_map.get(spin_clean, -1)
        if label == -1:
            continue  # 忽略无效 Spin

        # 开始和结束时间（秒）
        try:
            start_time = float(row["Begin Time - ss.msec"])
            end_time   = float(row["End Time - ss.msec"])
        except ValueError:
            # print(f"Warning: invalid time in line: {line}")
            continue

        # 构建 sample
        samples.append({
            "video_id": video_id,
            "video_path": str(video_path),
            "start_time": start_time,   # ← 改这里
            "end_time": end_time,       # ← 改这里
            "label": label
        })

    print(f"Total valid samples: {len(samples)}")
    return samples

# =========================
# 主函数
# =========================
if __name__ == "__main__":
    samples = parse_txt_to_samples(ANNOTATION_TXT, VIDEO_ROOT)

    # 保存为 samples.pt
    torch.save(samples, OUTPUT_SAMPLES)
    print(f"Saved {len(samples)} samples to {OUTPUT_SAMPLES}")
