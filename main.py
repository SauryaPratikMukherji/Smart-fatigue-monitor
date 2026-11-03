# FocusGuard - Desktop HUD Application
import sys
import time
import cv2
import numpy as np

from src.config import (
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_EAR_FRAMES_DROWSY,
    DEFAULT_MAR_THRESHOLD,
    DEFAULT_MAR_FRAMES_YAWN
)
from src.face_mesh_detector import FaceMeshDetector
from src.fatigue_metrics import (
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
    BlinkTracker,
    YawnTracker
)
from src.attention_scorer import AttentionScorer
from src.neural_classifier import AttentionNeuralClassifier
from src.break_recommender import BreakRecommender
from src.audio_alert import AlertManager
from src.session_logger import SessionLogger


# run the desktop opencv interface
def run_desktop_app(camera_index: int = 0):
    print("=" * 60)
    print("  FocusGuard - Student Attention & Fatigue Monitor")
    print("  College Python Project")
    print("=" * 60)
    print("[INFO] Initializing modules...")

    # initialize detectors and trackers
    detector = FaceMeshDetector()
    blink_tracker = BlinkTracker()
    yawn_tracker = YawnTracker()
    scorer = AttentionScorer()
    classifier = AttentionNeuralClassifier()
    recommender = BreakRecommender()
    alert_mgr = AlertManager(enabled=True)
    logger = SessionLogger()

    # open webcam
    print(f"[INFO] Opening camera #{camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        print("[TIP] Try changing camera index or plugging in webcam.")
        return

    # set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[READY] FocusGuard running.")
    print("Controls:")
    print("  'q' -> Exit and view summary")
    print("  'r' -> Reset counters")
    print("  'm' -> Toggle audio alert\n")

    fps_last_time = time.time()
    fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture failed.")
                break

            # calculate fps
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, now - fps_last_time))
            fps_last_time = now

            img_h, img_w = frame.shape[:2]

            # detect face mesh landmarks
            extracted = detector.process_frame(frame)

            if extracted is not None:
                # compute ear for both eyes
                ear_l = calculate_ear(extracted['left_eye_pts'])
                ear_r = calculate_ear(extracted['right_eye_pts'])
                ear_avg = (ear_l + ear_r) / 2.0
                mar = calculate_mar(extracted['mouth_pts'])

                # head pose angles
                pitch, yaw, roll, nose_proj = estimate_head_pose(
                    extracted['landmarks_pixel'], img_w, img_h
                )

                # update trackers
                is_blink, is_drowsy, total_blinks, bpm = blink_tracker.update(ear_avg)
                is_yawn, total_yawns = yawn_tracker.update(mar)

                # neural network prediction
                closure_ratio = blink_tracker.closed_frame_count / float(DEFAULT_EAR_FRAMES_DROWSY)
                yawn_ratio = yawn_tracker.open_mouth_frames / float(DEFAULT_MAR_FRAMES_YAWN)
                neural_state, confidence, _ = classifier.predict(
                    ear_l, ear_r, ear_avg, mar,
                    yaw, pitch, roll, bpm,
                    closure_ratio, yawn_ratio
                )

                # calculate final score and state
                attention_score, final_state = scorer.update(
                    has_face=True,
                    ear_avg=ear_avg,
                    is_prolonged_closure=is_drowsy,
                    mar=mar,
                    is_yawning=is_yawn,
                    yaw=yaw,
                    pitch=pitch,
                    bpm=bpm,
                    ear_thresh=DEFAULT_EAR_THRESHOLD,
                    neural_state=neural_state
                )

                # trigger beep if drowsy
                if is_drowsy:
                    alert_mgr.trigger_alert("drowsy")

                # draw annotations and hud
                display_frame = detector.draw_annotations(
                    frame, extracted, pitch, yaw, roll, nose_proj,
                    final_state, ear_avg, mar, attention_score
                )

                # record entry
                logger.log_entry(
                    now - scorer.start_time,
                    ear_avg, mar, yaw, pitch, bpm, attention_score, final_state
                )

            else:
                # no face in frame
                attention_score, final_state = scorer.update(
                    has_face=False,
                    ear_avg=0.0,
                    is_prolonged_closure=False,
                    mar=0.0,
                    is_yawning=False,
                    yaw=0.0,
                    pitch=0.0,
                    bpm=0.0,
                    ear_thresh=DEFAULT_EAR_THRESHOLD
                )
                display_frame = frame.copy()
                cv2.putText(
                    display_frame, "NO FACE DETECTED - PLEASE FACE CAMERA",
                    (30, int(img_h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

            # draw bottom status bar
            summary = scorer.get_summary()
            ribbon_text = f"FPS: {fps:.1f} | Blinks: {blink_tracker.total_blinks} ({bpm:.0f}/min) | Yawns: {yawn_tracker.total_yawns} | Time: {summary['session_duration_min']:.1f}m"
            cv2.rectangle(display_frame, (0, img_h - 30), (img_w, img_h), (15, 15, 15), -1)
            cv2.putText(display_frame, ribbon_text, (10, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            # show frame in opencv window
            cv2.imshow("FocusGuard - Attention Monitor", display_frame)

            # handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                scorer.reset()
                blink_tracker.reset()
                yawn_tracker.reset()
                logger.clear()
                print("[INFO] Session reset.")
            elif key == ord('m'):
                alert_mgr.set_enabled(not alert_mgr.enabled)
                status_str = "ENABLED" if alert_mgr.enabled else "MUTED"
                print(f"[INFO] Audio Alarm: {status_str}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    # print summary report on exit
    print("\n" + "=" * 60)
    print("               STUDY SESSION REPORT")
    print("=" * 60)
    sum_data = scorer.get_summary()
    print(f"  * Duration          : {sum_data['session_duration_min']:.1f} min ({sum_data['session_duration_sec']:.0f} sec)")
    print(f"  * Average Focus     : {sum_data['avg_attention']:.1f} %")
    print(f"  * Focused Time      : {sum_data['focused_pct']:.1f} %")
    print(f"  * Distracted Time   : {sum_data['distracted_pct']:.1f} %")
    print(f"  * Drowsy Time       : {sum_data['drowsy_pct']:.1f} %")
    print(f"  * Total Blinks      : {blink_tracker.total_blinks}")
    print(f"  * Total Yawns       : {yawn_tracker.total_yawns}")

    # export csv
    csv_file = logger.export_csv()
    print(f"\n[INFO] Session report saved: {csv_file}")
    
    # recommendations based on session statistics
    total_time_min = max(0.1, sum_data['session_duration_min'])
    avg_bpm = round(blink_tracker.total_blinks / total_time_min, 1)
    suggestions = recommender.generate_session_suggestions(
        summary=sum_data,
        total_blinks=blink_tracker.total_blinks,
        avg_bpm=avg_bpm,
        total_yawns=yawn_tracker.total_yawns,
        drowsy_episodes=blink_tracker.drowsy_episode_count
    )

    print("\n" + "-" * 60)
    print("        STUDY SUGGESTIONS & RECOMMENDATIONS")
    print("-" * 60)
    for item in suggestions:
        print(f"\n[{item['category']}] - {item['title']} ({item['level']})")
        print(f"  * Insight : {item['insight']}")
        print(f"  * Advice  : {item['actionable_advice']}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    cam_id = 0
    if len(sys.argv) > 1:
        try:
            cam_id = int(sys.argv[1])
        except ValueError:
            pass
    run_desktop_app(cam_id)
