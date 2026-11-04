# fatigue metrics: ear, mar, head pose, blink, yawn, and sleep tracking
import time
import math
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import src.config as cfg
import importlib
try:
    importlib.reload(cfg)
except Exception:
    pass

LEFT_EYE_LANDMARKS = cfg.LEFT_EYE_LANDMARKS
RIGHT_EYE_LANDMARKS = cfg.RIGHT_EYE_LANDMARKS
MOUTH_INNER_LIPS = cfg.MOUTH_INNER_LIPS
POSE_LANDMARK_INDICES = cfg.POSE_LANDMARK_INDICES
MODEL_3D_POINTS = cfg.MODEL_3D_POINTS
DEFAULT_EAR_THRESHOLD = getattr(cfg, 'DEFAULT_EAR_THRESHOLD', 0.22)
DEFAULT_EAR_FRAMES_DROWSY = getattr(cfg, 'DEFAULT_EAR_FRAMES_DROWSY', 25)
DEFAULT_EAR_FRAMES_BLINK = getattr(cfg, 'DEFAULT_EAR_FRAMES_BLINK', 2)
DEFAULT_MAR_THRESHOLD = getattr(cfg, 'DEFAULT_MAR_THRESHOLD', 0.58)
DEFAULT_MAR_FRAMES_YAWN = getattr(cfg, 'DEFAULT_MAR_FRAMES_YAWN', 25)
HEAD_YAW_THRESHOLD = getattr(cfg, 'HEAD_YAW_THRESHOLD', 25.0)
HEAD_PITCH_DOWN_THRESHOLD = getattr(cfg, 'HEAD_PITCH_DOWN_THRESHOLD', 18.0)
HEAD_PITCH_UP_THRESHOLD = getattr(cfg, 'HEAD_PITCH_UP_THRESHOLD', 22.0)
DEFAULT_SLEEP_TIMEOUT_SECONDS = getattr(cfg, 'DEFAULT_SLEEP_TIMEOUT_SECONDS', 120)
BLINK_WINDOW_SECONDS = getattr(cfg, 'BLINK_WINDOW_SECONDS', 60)


# euclidean distance helper
def euclidean_dist(pt1: np.ndarray, pt2: np.ndarray) -> float:
    return float(np.linalg.norm(pt1 - pt2))


# calculate eye aspect ratio
def calculate_ear(eye_points: np.ndarray) -> float:
    if len(eye_points) < 6:
        return 0.30

    d_v1 = euclidean_dist(eye_points[1], eye_points[5])
    d_v2 = euclidean_dist(eye_points[2], eye_points[4])
    d_h = euclidean_dist(eye_points[0], eye_points[3])

    if d_h == 0:
        return 0.0

    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return ear


# calculate mouth aspect ratio
def calculate_mar(lip_dict_points: Dict[str, np.ndarray]) -> float:
    try:
        top = lip_dict_points['top']
        bottom = lip_dict_points['bottom']
        left = lip_dict_points['left_corner']
        right = lip_dict_points['right_corner']
        top_left = lip_dict_points.get('top_left', top)
        bottom_left = lip_dict_points.get('bottom_left', bottom)
        top_right = lip_dict_points.get('top_right', top)
        bottom_right = lip_dict_points.get('bottom_right', bottom)

        v_center = euclidean_dist(top, bottom)
        v_left = euclidean_dist(top_left, bottom_left)
        v_right = euclidean_dist(top_right, bottom_right)
        h_span = euclidean_dist(left, right)

        if h_span == 0:
            return 0.0

        mar = (v_center + v_left + v_right) / (3.0 * h_span)
        return mar
    except KeyError:
        return 0.0


# estimate head pose angles using solvePnP
def estimate_head_pose(
    landmarks_pixel: np.ndarray,
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, np.ndarray]:
    default_nose = (int(landmarks_pixel[POSE_LANDMARK_INDICES[0]][0]),
                    int(landmarks_pixel[POSE_LANDMARK_INDICES[0]][1]))

    try:
        image_points = np.array([
            landmarks_pixel[POSE_LANDMARK_INDICES[0]][:2],
            landmarks_pixel[POSE_LANDMARK_INDICES[1]][:2],
            landmarks_pixel[POSE_LANDMARK_INDICES[2]][:2],
            landmarks_pixel[POSE_LANDMARK_INDICES[3]][:2],
            landmarks_pixel[POSE_LANDMARK_INDICES[4]][:2],
            landmarks_pixel[POSE_LANDMARK_INDICES[5]][:2],
        ], dtype=np.float64)

        model_points = np.array(MODEL_3D_POINTS, dtype=np.float64)

        focal_length = float(img_w)
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, default_nose

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0] * 360.0)
        yaw = float(angles[1] * 360.0)
        roll = float(angles[2] * 360.0)

        axis_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
        nose_proj, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        nose_proj_point = (int(nose_proj[0][0][0]), int(nose_proj[0][0][1]))

        return pitch, yaw, roll, nose_proj_point
    except Exception:
        return 0.0, 0.0, 0.0, default_nose


# robust blink tracker with calibration and noise rejection
class BlinkTracker:
    def __init__(self, ear_threshold: float = DEFAULT_EAR_THRESHOLD, drowsy_frames: int = DEFAULT_EAR_FRAMES_DROWSY):
        self.ear_threshold = ear_threshold
        self.drowsy_frames = drowsy_frames
        self.closed_frame_count = 0
        self.total_blinks = 0
        self.blink_timestamps: List[float] = []
        self.is_prolonged_closure = False
        self.drowsy_episode_count = 0

        # internal tracking states
        self._was_closed = False
        self._closure_start_time = 0.0
        self._last_blink_time = 0.0
        self._warmup_frames = 20
        self._frame_count = 0

        # baseline tracking for user's open eyes
        self.open_eye_baseline = 0.29

    def update(self, ear: float) -> Tuple[bool, bool, int, float]:
        current_time = time.time()
        is_blink_detected = False
        self._frame_count += 1

        # warmup period: calibrate baseline without counting blinks
        if self._frame_count <= self._warmup_frames:
            if ear >= 0.18:
                self.open_eye_baseline = (0.85 * self.open_eye_baseline) + (0.15 * ear)
            return False, False, 0, 0.0

        # compute dynamic closure threshold based on user's open eyes
        effective_threshold = max(0.16, min(self.ear_threshold, self.open_eye_baseline * 0.72))
        recovery_threshold = effective_threshold + 0.035

        # detect eye closure
        if ear < effective_threshold:
            if not self._was_closed:
                self._was_closed = True
                self._closure_start_time = current_time
            self.closed_frame_count += 1

            closed_duration = current_time - self._closure_start_time
            if self.closed_frame_count >= self.drowsy_frames or closed_duration >= 1.2:
                if not self.is_prolonged_closure:
                    self.is_prolonged_closure = True
                    self.drowsy_episode_count += 1
            else:
                self.is_prolonged_closure = False

        # detect eye opening
        elif ear >= recovery_threshold:
            if self._was_closed:
                closed_duration = current_time - self._closure_start_time
                time_since_last_blink = current_time - self._last_blink_time

                # valid blink: at least 2 frames, duration between 0.07s and 0.75s, refractory debounce >= 0.22s
                if (self.closed_frame_count >= 2) and (0.07 <= closed_duration <= 0.75) and (time_since_last_blink >= 0.22):
                    self.total_blinks += 1
                    self.blink_timestamps.append(current_time)
                    self._last_blink_time = current_time
                    is_blink_detected = True

                self.closed_frame_count = 0
                self._was_closed = False
                self.is_prolonged_closure = False
            else:
                # slowly adapt open-eye baseline when eyes are stably open
                if ear > effective_threshold + 0.04:
                    self.open_eye_baseline = (0.98 * self.open_eye_baseline) + (0.02 * ear)

        # clean blinks older than 60s
        cutoff_time = current_time - BLINK_WINDOW_SECONDS
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff_time]

        # calculate rolling BPM
        if len(self.blink_timestamps) == 0:
            bpm = 0.0
        else:
            time_span = min(BLINK_WINDOW_SECONDS, max(5.0, current_time - self.blink_timestamps[0]))
            bpm = (len(self.blink_timestamps) / time_span) * 60.0

        return is_blink_detected, self.is_prolonged_closure, self.total_blinks, round(bpm, 1)

    def reset(self):
        self.closed_frame_count = 0
        self.total_blinks = 0
        self.blink_timestamps.clear()
        self.is_prolonged_closure = False
        self.drowsy_episode_count = 0
        self._was_closed = False
        self._closure_start_time = 0.0
        self._last_blink_time = 0.0
        self._frame_count = 0
        self.open_eye_baseline = 0.29


# class to track yawns
class YawnTracker:
    def __init__(self, mar_threshold: float = DEFAULT_MAR_THRESHOLD, yawn_frames: int = DEFAULT_MAR_FRAMES_YAWN):
        self.mar_threshold = mar_threshold
        self.yawn_frames = yawn_frames
        self.open_mouth_frames = 0
        self.total_yawns = 0
        self.is_yawning = False
        self._yawn_in_progress = False

    def update(self, mar: float) -> Tuple[bool, int]:
        if mar > self.mar_threshold:
            self.open_mouth_frames += 1
            if self.open_mouth_frames >= self.yawn_frames:
                self.is_yawning = True
                if not self._yawn_in_progress:
                    self.total_yawns += 1
                    self._yawn_in_progress = True
        else:
            self.open_mouth_frames = 0
            self.is_yawning = False
            self._yawn_in_progress = False

        return self.is_yawning, self.total_yawns

    def reset(self):
        self.open_mouth_frames = 0
        self.total_yawns = 0
        self.is_yawning = False
        self._yawn_in_progress = False


# tracker for student falling asleep (continuous eye closure or head down > 2 minutes)
class SleepTracker:
    def __init__(self, timeout_seconds: float = DEFAULT_SLEEP_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds
        self.eyes_closed_start: Optional[float] = None
        self.head_down_start: Optional[float] = None
        self.eyes_closed_duration: float = 0.0
        self.head_down_duration: float = 0.0
        self.alarm_triggered: bool = False
        self.alarm_dismissed: bool = False
        self.trigger_reason: str = ""

    def update(
        self,
        is_eyes_closed: bool,
        is_head_down: bool,
        current_time: float
    ) -> Tuple[bool, str, float, float]:
        # track continuous eye closure
        if is_eyes_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            self.eyes_closed_duration = current_time - self.eyes_closed_start
        else:
            self.eyes_closed_start = None
            self.eyes_closed_duration = 0.0

        # track continuous head down
        if is_head_down:
            if self.head_down_start is None:
                self.head_down_start = current_time
            self.head_down_duration = current_time - self.head_down_start
        else:
            self.head_down_start = None
            self.head_down_duration = 0.0

        # check if 2 minutes timeout reached
        if not self.alarm_triggered and not self.alarm_dismissed:
            if self.eyes_closed_duration >= self.timeout_seconds:
                self.alarm_triggered = True
                mins = self.eyes_closed_duration / 60.0
                self.trigger_reason = f"Eyes closed for {mins:.1f} min"
            elif self.head_down_duration >= self.timeout_seconds:
                self.alarm_triggered = True
                mins = self.head_down_duration / 60.0
                self.trigger_reason = f"Head down on desk for {mins:.1f} min"

        # reset dismissed flag once student wakes up (eyes open and head up)
        if self.alarm_dismissed:
            if not is_eyes_closed and not is_head_down:
                self.alarm_dismissed = False

        return self.alarm_triggered, self.trigger_reason, self.eyes_closed_duration, self.head_down_duration

    # manual stop by student
    def dismiss(self):
        self.alarm_triggered = False
        self.alarm_dismissed = True
        self.eyes_closed_start = None
        self.head_down_start = None
        self.eyes_closed_duration = 0.0
        self.head_down_duration = 0.0
        self.trigger_reason = ""

    def reset(self):
        self.eyes_closed_start = None
        self.head_down_start = None
        self.eyes_closed_duration = 0.0
        self.head_down_duration = 0.0
        self.alarm_triggered = False
        self.alarm_dismissed = False
        self.trigger_reason = ""
