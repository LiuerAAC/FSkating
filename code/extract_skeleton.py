# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# # ---- 配置 ----
# # 选一个你已经有的视频路径
# VIDEO_PATH = "/Users/yuxuancao/Desktop/GraduateDesign/code/Dataset/0.mp4"
# OUTPUT_PATH = "pose_debug_output.mp4"

# # 初始化 MediaPipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1, # 1 为平衡，2 最准但慢
#     enable_segmentation=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def calculate_angle_2d(p1, p2):
#     """计算两点连线与水平轴的角度 (度)"""
#     return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

# def main():
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

#     print(f"[INFO] 正在处理视频: {VIDEO_PATH}")
    
#     prev_angle = None
#     total_rotation = 0.0

#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             break

#         # 转换为 RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = pose.process(image_rgb)

#         if results.pose_landmarks:
#             # 1. 绘制骨架
#             mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             # 2. 提取关键点 (以肩膀为例)
#             landmarks = results.pose_landmarks.landmark
#             l_shld = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
#             r_shld = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

#             # 转换为像素坐标
#             p1 = (int(l_shld.x * width), int(l_shld.y * height))
#             p2 = (int(r_shld.x * width), int(r_shld.y * height))

#             # 3. 计算旋转角度
#             # 注意：在 2D 旋转中，单纯靠角度会遇到“左右互换”导致的角度突变
#             # 这里的简单演示仅展示角度捕捉，正式周数计算需结合 MediaPipe 的 Z 轴
#             current_angle = calculate_angle_2d(p1, p2)
            
#             # 显示当前角度
#             cv2.putText(image, f"Angle: {current_angle:.1f}", (50, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
#             # 打印 Z 轴深度（判断前后关系的关键）
#             cv2.putText(image, f"L_Z: {l_shld.z:.3f}", (50, 90), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         out.write(image)
#         cv2.imshow('MediaPipe Pose Test', image)
#         if cv2.waitKey(5) & 0xFF == 27: # 按 ESC 退出
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"[OK] 处理完成，结果已保存至: {OUTPUT_PATH}")

# if __name__ == "__main__":
#     main()

import cv2
import mediapipe as mp
import json
import os

# ---- 配置 ----
JSONL_INPUT = "/Users/yuxuancao/Desktop/GraduateDesign/code/spins_with_buffer1.jsonl"
SAVE_DIR = "/Users/yuxuancao/Desktop/GraduateDesign/code/skeleton"
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