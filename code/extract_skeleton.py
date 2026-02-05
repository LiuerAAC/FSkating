import cv2
import mediapipe as mp
import json
import os

# ---- 配置 ----
JSONL_INPUT = "GraduateDesign/code/spins_with_buffer1.jsonl"
SAVE_DIR = "GraduateDesign/code/skeleton"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

def extract_segments():
    with open(JSONL_INPUT, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            video_path = data['video_path']
            video_id = data['video_id']
            spin_id = data['spin_id']
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # 每个 spin 存储一个独立的 json
            spin_skeleton = {
                "video_id": video_id,
                "spin_id": spin_id,
                "segments_data": []
            }

            print(f"正在处理 Video {video_id} - Spin {spin_id}...")

            for seg in data['segments']:
                label = seg['label']
                start_frame = int(seg['begin'] * fps)
                end_frame = int(seg['end'] * fps)
                
                seg_frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for f_idx in range(start_frame, end_frame + 1):
                    success, frame = cap.read()
                    if not success: break
                    
                    # 每一帧提取关键点
                    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if res.pose_landmarks:
                        lm = res.pose_landmarks.landmark
                        # 仅保存旋转分析的核心点，进一步压缩体积
                        # 肩(11,12), 胯(23,24), 膝(25,26)
                        points = {}
                        for i in [11, 12, 23, 24, 25, 26]:
                            points[i] = {"x": round(lm[i].x, 6), "y": round(lm[i].y, 6)}
                        seg_frames.append({"f": f_idx, "pts": points})
                
                spin_skeleton["segments_data"].append({
                    "label": label,
                    "frames": seg_frames
                })

            cap.release()

            # 保存文件名：v1_spin1_skeleton.json
            out_name = f"v{video_id}_s{spin_id}_skeleton.json"
            with open(os.path.join(SAVE_DIR, out_name), 'w') as f:
                json.dump(spin_skeleton, f)
            print(f"  [完成] 骨骼片段已存至: {out_name}")

if __name__ == "__main__":
    extract_segments()