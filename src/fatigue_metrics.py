import time
import math
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from src.config import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    MOUTH_INNER_LIPS,
    POSE_LANDMARK_INDICES,
    MODEL_3D_POINTS,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_EAR_FRAMES_DROWSY,
    DEFAULT_EAR_FRAMES_BLINK,
    DEFAULT_MAR_THRESHOLD,
    DEFAULT_MAR_FRAMES_YAWN,
    HEAD_YAW_THRESHOLD,
    HEAD_PITCH_DOWN_THRESHOLD,
    HEAD_PITCH_UP_THRESHOLD,
    BLINK_WINDOW_SECONDS
)


# euclidean distance helper
def euclidean_dist(pt1: np.ndarray, pt2: np.ndarray) -> float:
    return float(np.linalg.norm(pt1 - pt2))


# calculate eye aspect ratio
def calculate_ear(eye_points: np.ndarray) -> float:
    if len(eye_points) < 6:
        return 0.30

    # vertical distances
    d_v1 = euclidean_dist(eye_points[1], eye_points[5])
    d_v2 = euclidean_dist(eye_points[2], eye_points[4])

    # horizontal distance
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

        # vertical distances
        v_center = euclidean_dist(top, bottom)
        v_left = euclidean_dist(top_left, bottom_left)
        v_right = euclidean_dist(top_right, bottom_right)

        # horizontal distance
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
        # 2D points on image
        image_points = np.array([
            landmarks_pixel[POSE_LANDMARK_INDICES[0]][:2],  # nose
            landmarks_pixel[POSE_LANDMARK_INDICES[1]][:2],  # chin
            landmarks_pixel[POSE_LANDMARK_INDICES[2]][:2],  # left eye
            landmarks_pixel[POSE_LANDMARK_INDICES[3]][:2],  # right eye
            landmarks_pixel[POSE_LANDMARK_INDICES[4]][:2],  # left mouth
            landmarks_pixel[POSE_LANDMARK_INDICES[5]][:2],  # right mouth
        ], dtype=np.float64)

        model_points = np.array(MODEL_3D_POINTS, dtype=np.float64)

        # camera matrix
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

        # get rotation matrix and euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0] * 360.0)
        yaw = float(angles[1] * 360.0)
        roll = float(angles[2] * 360.0)

        # nose projection point for direction arrow
        axis_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
        nose_proj, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        nose_proj_point = (int(nose_proj[0][0][0]), int(nose_proj[0][0][1]))

        return pitch, yaw, roll, nose_proj_point
    except Exception:
        return 0.0, 0.0, 0.0, default_nose


# class to track blinks and calculate blink rate
class BlinkTracker:
    def __init__(self, ear_threshold: float = DEFAULT_EAR_THRESHOLD, drowsy_frames: int = DEFAULT_EAR_FRAMES_DROWSY):
        self.ear_threshold = ear_threshold
        self.drowsy_frames = drowsy_frames
        self.closed_frame_count = 0
        self.total_blinks = 0
        self.blink_timestamps: List[float] = []
        self.is_prolonged_closure = False
        self.drowsy_episode_count = 0
        self._was_closed = False
        self._closure_start_time = 0.0
        self._last_blink_time = 0.0

        # baseline tracking for open eyes
        self.open_eye_baseline = 0.31

    def update(self, ear: float) -> Tuple[bool, bool, int, float]:
        current_time = time.time()
        is_blink_detected = False

        # update open eye baseline when eyes are open
        if ear >= 0.25:
            self.open_eye_baseline = (0.97 * self.open_eye_baseline) + (0.03 * ear)

        # threshold calculation
        effective_threshold = max(0.18, min(self.ear_threshold, self.open_eye_baseline * 0.78))
        recovery_threshold = effective_threshold + 0.02

        if ear < effective_threshold:
            if not self._was_closed:
                self._was_closed = True
                self._closure_start_time = current_time
            self.closed_frame_count += 1

            # check if closed long enough to count as drowsy
            closed_duration = current_time - self._closure_start_time
            if self.closed_frame_count >= self.drowsy_frames or closed_duration >= 1.1:
                if not self.is_prolonged_closure:
                    self.is_prolonged_closure = True
                    self.drowsy_episode_count += 1
            else:
                self.is_prolonged_closure = False

        elif ear >= recovery_threshold or (self._was_closed and ear > effective_threshold):
            # eye opened back up
            if self._was_closed:
                closed_duration = current_time - self._closure_start_time
                time_since_last_blink = current_time - self._last_blink_time

                # count as blink if closed briefly and not spamming
                if (1 <= self.closed_frame_count < self.drowsy_frames) and (closed_duration < 0.95) and (time_since_last_blink > 0.14):
                    self.total_blinks += 1
                    self.blink_timestamps.append(current_time)
                    self._last_blink_time = current_time
                    is_blink_detected = True

                self.closed_frame_count = 0
                self._was_closed = False
                self.is_prolonged_closure = False

        # remove old blinks past 60 seconds
        cutoff_time = current_time - BLINK_WINDOW_SECONDS
        self.blink_timestamps = [t for t in self.blink_timestamps if t >= cutoff_time]

        # calculate blinks per minute
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
        self.open_eye_baseline = 0.31


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
        # check if mouth is open
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
