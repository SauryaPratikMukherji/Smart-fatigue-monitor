import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import mediapipe as mp

from src.config import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    MOUTH_INNER_LIPS,
    POSE_LANDMARK_INDICES
)


class FaceMeshDetector:
    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    # process image frame and return landmarks
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        img_h, img_w = frame_bgr.shape[:2]

        # convert to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # convert normalized coordinates to pixel values
        landmarks_pixel = np.array([
            [int(lm.x * img_w), int(lm.y * img_h), lm.z]
            for lm in face_landmarks.landmark
        ], dtype=np.float64)

        # left eye points
        left_eye_pts = np.array([
            landmarks_pixel[idx][:2] for idx in LEFT_EYE_LANDMARKS
        ], dtype=np.float64)

        # right eye points
        right_eye_pts = np.array([
            landmarks_pixel[idx][:2] for idx in RIGHT_EYE_LANDMARKS
        ], dtype=np.float64)

        # mouth points
        mouth_pts = {
            key: landmarks_pixel[idx][:2]
            for key, idx in MOUTH_INNER_LIPS.items()
        }

        # face bounding box
        x_min = int(max(0, np.min(landmarks_pixel[:, 0])))
        y_min = int(max(0, np.min(landmarks_pixel[:, 1])))
        x_max = int(min(img_w, np.max(landmarks_pixel[:, 0])))
        y_max = int(min(img_h, np.max(landmarks_pixel[:, 1])))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        return {
            'landmarks_pixel': landmarks_pixel,
            'left_eye_pts': left_eye_pts,
            'right_eye_pts': right_eye_pts,
            'mouth_pts': mouth_pts,
            'bbox': bbox,
            'raw_landmarks': face_landmarks
        }

    # draw face box, eye lines, direction arrow, and info box on frame
    def draw_annotations(
        self,
        frame: np.ndarray,
        extracted: Dict[str, Any],
        pitch: float,
        yaw: float,
        roll: float,
        nose_proj_point: Tuple[int, int],
        state_label: str,
        ear_avg: float,
        mar: float,
        attention_score: float
    ) -> np.ndarray:
        annotated = frame.copy()
        img_h, img_w = annotated.shape[:2]
        landmarks_pixel = extracted['landmarks_pixel']
        bbox = extracted['bbox']

        # set color by state
        if state_label == "Focused":
            theme_color = (0, 220, 0)
        elif state_label == "Distracted":
            theme_color = (0, 165, 255)
        elif state_label == "Drowsy":
            theme_color = (0, 0, 255)
        else:
            theme_color = (180, 180, 180)

        # 1. draw bounding box
        x, y, w, h = bbox
        corner_len = int(min(w, h) * 0.15)
        thick = 2

        cv2.line(annotated, (x, y), (x + corner_len, y), theme_color, thick)
        cv2.line(annotated, (x, y), (x, y + corner_len), theme_color, thick)
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), theme_color, thick)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), theme_color, thick)
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), theme_color, thick)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), theme_color, thick)
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), theme_color, thick)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), theme_color, thick)

        # 2. draw eye contours
        left_contour_pts = np.array([landmarks_pixel[i][:2] for i in LEFT_EYE_CONTOUR], dtype=np.int32)
        right_contour_pts = np.array([landmarks_pixel[i][:2] for i in RIGHT_EYE_CONTOUR], dtype=np.int32)
        cv2.polylines(annotated, [left_contour_pts], True, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.polylines(annotated, [right_contour_pts], True, (255, 255, 0), 1, cv2.LINE_AA)

        # 3. draw mouth contour
        mouth_inner_pts = np.array([
            landmarks_pixel[i][:2] for i in [78, 82, 13, 312, 308, 317, 14, 87]
        ], dtype=np.int32)
        cv2.polylines(annotated, [mouth_inner_pts], True, (200, 100, 255), 1, cv2.LINE_AA)

        # 4. draw head direction arrow from nose
        try:
            nose_pt = (int(landmarks_pixel[1][0]), int(landmarks_pixel[1][1]))
            safe_proj = (
                int(np.clip(nose_proj_point[0], -500, img_w + 500)),
                int(np.clip(nose_proj_point[1], -500, img_h + 500))
            )
            cv2.arrowedLine(annotated, nose_pt, safe_proj, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.2)
        except Exception:
            pass

        # 5. draw info box overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (320, 140), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        # text labels
        cv2.putText(annotated, f"Status: {state_label.upper()}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, theme_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Attention: {attention_score:.1f}%", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"EAR: {ear_avg:.2f}  |  MAR: {mar:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Yaw: {yaw:+.1f} | Pitch: {pitch:+.1f}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 220), 1, cv2.LINE_AA)

        return annotated

    def close(self):
        self.face_mesh.close()
