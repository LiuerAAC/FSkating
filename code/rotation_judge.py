import json
import numpy as np
import os


class SkeletonAnalyzer:
    def __init__(self, skeleton_json_path):
        self.observation_points = [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
        self.fps = 30
        self.default_avg_speed = 0.065
        self.skeleton_data = self._load_data(skeleton_json_path)
        # Get base frame data
        self.frames_list = self._extract_frames(self.skeleton_data)

    def _load_data(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Skeleton file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_frames(self, data):
        if isinstance(data, list): return data
        if 'segments_data' in data:
            all_f = []
            for s in data['segments_data']: all_f.extend(s.get('frames', []))
            return sorted(all_f, key=lambda x: x['f'])
        return data.get('frames', [])

    def _get_point(self, pts, idx):
        p = pts.get(str(idx)) or pts.get(idx)
        if p is None: return None
        return np.array([p["x"], p["y"], p.get("z", 0.0)], dtype=np.float32)

    def get_frame_support_leg(self, pts):
        """
        Core algorithm replacement: Decision logic based on the angle between the spatial line and Y-axis.
        Calculates the cosine value between the vector (Hip -> Ankle) and the Y-axis (0, 1, 0).
        A larger absolute cosine value indicates a smaller angle (more aligned with the vertical axis),
        resulting in a 'support leg' determination.
        """
        try:
            # Get keypoints: 23-L_Hip, 27-L_Ankle, 24-R_Hip, 28-R_Ankle
            # Use _get_point to safely handle both int and str keys in pts
            l_hip = self._get_point(pts, 23)
            l_ank = self._get_point(pts, 27)
            r_hip = self._get_point(pts, 24)
            r_ank = self._get_point(pts, 28)

            if l_hip is None or l_ank is None or r_hip is None or r_ank is None:
                return None

            # Calculate left and right leg vectors
            vec_l = l_ank - l_hip
            vec_r = r_ank - r_hip

            def get_vertical_cos(v):
                mag = np.linalg.norm(v)
                if mag == 0: return 0
                # The dot product of v and (0,1,0) is simply v[1]
                # Take absolute value because both 0 or 180 degrees represent vertical alignment
                return abs(v[1] / mag)

            cos_l = get_vertical_cos(vec_l)
            cos_r = get_vertical_cos(vec_r)

            # Get confidence weights ('v' field in MediaPipe)
            # Compatible with both dictionary and array formats
            l_v = pts.get('27', {}).get('v', 0.5) if isinstance(pts.get('27'), dict) else 0.5
            r_v = pts.get('28', {}).get('v', 0.5) if isinstance(pts.get('28'), dict) else 0.5
            weight = (l_v + r_v) / 2

            return ("Left", weight) if cos_l > cos_r else ("Right", weight)

        except Exception:
            return None

    def calculate_revs_for_segment(self, frames, support_leg):
        frame_count = len(frames)
        if frame_count < 5: return 0.0

        center_idx = 27 if support_leg == "Left" else 28
        all_speeds = []

        for kp in self.observation_points:
            if kp == center_idx: continue
            angles = []
            for f in frames:
                pts = f.get("pts", {})
                c = self._get_point(pts, center_idx)
                p = self._get_point(pts, kp)
                if c is None or p is None: continue
                v = p - c
                # Calculate rotation angle using the XZ plane
                angles.append(np.arctan2(v[0], v[2]))

            if len(angles) < 2: continue
            unwrapped = np.unwrap(np.array(angles))
            total_revs = abs(unwrapped[-1] - unwrapped[0]) / (2 * np.pi)
            all_speeds.append(total_revs / len(angles))

        if not all_speeds:
            return round(frame_count * self.default_avg_speed, 2)

        best_speed = max(all_speeds)
        # Empirical speed thresholds for rotation estimation
        if best_speed < 0.035:
            final_revs = frame_count * self.default_avg_speed
        elif best_speed > 0.16:
            final_revs = frame_count * 0.16
        else:
            final_revs = (frame_count * best_speed) + 0.5

        return round(final_revs, 2)

    def determine_rotation_direction(self, frames, support_leg):
        if len(frames) < 10: return "Unknown"
        center_idx = 27 if support_leg == "Left" else 28
        diffs = []
        for kp in [11, 12, 23, 24]:
            angles = []
            for f in frames:
                pts = f.get("pts", {})
                c = self._get_point(pts, center_idx)
                p = self._get_point(pts, kp)
                if c is not None and p is not None:
                    v = p - c
                    angles.append(np.arctan2(v[0], v[2]))
            if len(angles) >= 5:
                unwrapped = np.unwrap(np.array(angles))
                diffs.append(unwrapped[-1] - unwrapped[0])

        if not diffs: return "Unknown"
        avg_diff = np.mean(diffs)
        return "CCW" if avg_diff > 0.5 else ("CW" if avg_diff < -0.5 else "Unknown")

    def analyze_segment(self, start_time, end_time, label, target_foot="N/A", last_leg=None):
        # Timeline mapping
        start_idx = int(start_time * self.fps)
        end_idx = int(end_time * self.fps)
        frames = self.frames_list[start_idx: end_idx]

        if not frames:
            return {"leg": "Unknown", "revs": 0.0, "direction": "Unknown"}

        # Support leg voting mechanism
        votes = {"Left": 0.0, "Right": 0.0}
        for f in frames:
            res = self.get_frame_support_leg(f.get('pts', {}))
            if res:
                leg, weight = res
                votes[leg] += weight

        voted_leg = "Left" if votes["Left"] > votes["Right"] else "Right"

        # Change-of-foot logic (Forced flip for CP/JCP labels)
        if label in ["CP", "JCP"] and last_leg in ["Left", "Right"]:
            voted_leg = "Right" if last_leg == "Left" else "Left"

        direction = self.determine_rotation_direction(frames, voted_leg)
        revs = self.calculate_revs_for_segment(frames, voted_leg)

        return {
            "leg": voted_leg,
            "revs": revs,
            "direction": direction
        }
