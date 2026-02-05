import json
from pathlib import Path
from collections import Counter

# --- 配置参数 ---
BUFFER_SEC = 2.0  
VIDEO_EXT = ".mp4"
# 触发旋转组提取的关键标记
IN_MARKERS = ("Jump In", "Difficult In", "Step In")
OUT_MARKERS = ("Jump Out", "Difficult Out", "Step Out")

def parse_txt(txt_path: Path):
    by_file = {}
    if not txt_path.exists():
        print(f"[ERROR] 文件不存在: {txt_path}")
        return by_file

    with txt_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # 跳过标题行
    for line in lines[1:]:
        line = line.strip()
        if not line: continue
        parts = line.split('\t')
        
        # 适配 0to34.txt 的 9 列结构
        if len(parts) < 9: continue

        try:
            begin_sec = float(parts[1]) # Begin Time - ss.msec
            end_sec = float(parts[3])   # End Time - ss.msec
            # 提取 Spin 名，去掉 [a1] 等注释
            spin_col = parts[5].strip().split('[')[0].strip() 
            # 提取 Edge 名，去掉 [a1] 等注释
            edge_col = parts[7].strip().split('[')[0].strip() 
            eaf_file = parts[8].strip()

            # --- 标签清洗与逻辑处理 ---
            # 默认标签就是 Spin 列的内容
            full_label = spin_col
            
            # 仅当 Edge 列明确标注为 Inside 时才增加后缀
            # 忽略 Outside 或 None/空值
            if "Inside" in edge_col:
                full_label = f"{spin_col}+Inside"

            by_file.setdefault(eaf_file, []).append({
                "begin": begin_sec,
                "end": end_sec,
                "label": full_label
            })
        except (ValueError, IndexError):
            continue
    return by_file

def extract_spins(rows):
    """
    按照 In -> (Segments) -> Out 的逻辑提取整组旋转
    """
    spins = []
    current = None
    spin_id = 0

    for r in rows:
        label = r["label"]
        # 检测旋转组开始
        if any(m in label for m in IN_MARKERS):
            current = {
                "spin_id": spin_id,
                "start": r["begin"],
                "end": None,
                "segments": []
            }
        
        if current is not None:
            current["segments"].append({
                "label": label,
                "begin": r["begin"],
                "end": r["end"]
            })

            # 检测旋转组结束
            if any(m in label for m in OUT_MARKERS):
                current["end"] = r["end"]
                spins.append(current)
                spin_id += 1
                current = None 
    return spins

def main():
    # --- 路径配置 (请确保与你的实际路径一致) ---
    txt_path = Path("GraduateDesign/Dataset/0to34.txt")
    out_path = Path("GraduateDesign/code/spins_with_buffer_clean.jsonl")
    dataset_dir = Path("GraduateDesign/Dataset/video")
    
    by_file = parse_txt(txt_path)
    all_extracted = []

    with out_path.open("w", encoding="utf-8") as fout:
        for eaf_file in sorted(by_file.keys()):
            rows = by_file[eaf_file]
            video_id = eaf_file.replace(".eaf", "")
            video_path = dataset_dir / f"{video_id}{VIDEO_EXT}"

            spins = extract_spins(rows)
            for spin in spins:
                # 计算带 Buffer 的剪辑时间范围
                clip_start = max(0.0, spin["start"] - BUFFER_SEC)
                clip_end = spin["end"] + BUFFER_SEC
                
                # 构建输出结构
                output_data = {
                    "spin_id": spin["spin_id"],
                    "start": spin["start"],
                    "end": spin["end"],
                    "segments": spin["segments"],
                    "clip_start": round(clip_start, 3),
                    "clip_end": round(clip_end, 3),
                    "video_id": video_id,
                    "video_path": str(video_path)
                }
                
                fout.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                all_extracted.append(output_data)

    print(f"[OK] 成功提取 {len(all_extracted)} 组旋转动作。")
    print(f"[INFO] 默认旋转已简化（去除了+Outside），仅保留+Inside标注。")
    print(f"[SAVE] 结果已存入: {out_path}")

if __name__ == "__main__":
    main()