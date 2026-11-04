# FocusGuard - Web Dashboard
# Student Attention and Fatigue Monitoring System
import time
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import src.config as cfg
import importlib
try:
    importlib.reload(cfg)
except Exception:
    pass

DEFAULT_EAR_THRESHOLD = getattr(cfg, 'DEFAULT_EAR_THRESHOLD', 0.22)
DEFAULT_EAR_FRAMES_DROWSY = getattr(cfg, 'DEFAULT_EAR_FRAMES_DROWSY', 25)
DEFAULT_MAR_THRESHOLD = getattr(cfg, 'DEFAULT_MAR_THRESHOLD', 0.58)
DEFAULT_MAR_FRAMES_YAWN = getattr(cfg, 'DEFAULT_MAR_FRAMES_YAWN', 25)
HEAD_PITCH_DOWN_THRESHOLD = getattr(cfg, 'HEAD_PITCH_DOWN_THRESHOLD', 18.0)
DEFAULT_SLEEP_TIMEOUT_SECONDS = getattr(cfg, 'DEFAULT_SLEEP_TIMEOUT_SECONDS', 120)
from src.face_mesh_detector import FaceMeshDetector
from src.fatigue_metrics import (
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
    BlinkTracker,
    YawnTracker,
    SleepTracker
)
from src.attention_scorer import AttentionScorer
from src.neural_classifier import AttentionNeuralClassifier
from src.break_recommender import BreakRecommender
from src.audio_alert import AlertManager
from src.session_logger import SessionLogger

# page configuration
st.set_page_config(
    page_title="FocusGuard - Student Attention & Fatigue Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# light theme css styling
st.markdown("""
<style>
    /* light mode background */
    .stApp {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
    }

    /* sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #0F172A !important;
    }

    /* header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 1.2rem;
    }

    /* metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.0rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 4px;
    }

    /* status badges */
    .status-badge-focused {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .status-badge-distracted {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .status-badge-drowsy {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    .status-badge-away {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.75; }
    }

    /* button styling */
    div.stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #334155 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }

    /* video container */
    div[data-testid="stImage"] {
        background: #FFFFFF;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
</style>
""", unsafe_allow_html=True)


# generate recommendations based on session statistics
def get_personalized_suggestions(summary, total_blinks, avg_bpm, total_yawns, drowsy_episodes):
    suggestions = []
    duration_min = summary.get('session_duration_min', 0.0)
    avg_focus = summary.get('avg_attention', 0.0)
    focused_pct = summary.get('focused_pct', 0.0)
    distracted_pct = summary.get('distracted_pct', 0.0)
    drowsy_pct = summary.get('drowsy_pct', 0.0)

    # focus analysis
    if avg_focus >= 80.0 and focused_pct >= 75.0:
        suggestions.append({
            'category': 'Focus & Concentration',
            'icon': '🎯',
            'level': 'EXCELLENT',
            'badge_color': '#ECFDF5',
            'border_color': '#A7F3D0',
            'text_color': '#065F46',
            'title': 'High Focus Maintained',
            'insight': f"Average attention was {avg_focus:.1f}% with {focused_pct:.0f}% of session in focus.",
            'actionable_advice': "Good focus session. Keep study stints around 45-50 minutes, followed by a short 5-minute break."
        })
    elif avg_focus >= 60.0:
        suggestions.append({
            'category': 'Focus & Concentration',
            'icon': '🎯',
            'level': 'MODERATE',
            'badge_color': '#FFFBEB',
            'border_color': '#FDE68A',
            'text_color': '#92400E',
            'title': 'Moderate Focus',
            'insight': f"Average attention was {avg_focus:.1f}%, with {distracted_pct:.0f}% of time looking away.",
            'actionable_advice': "Try using the Pomodoro technique (25 min study / 5 min break) to keep attention sharp."
        })
    else:
        suggestions.append({
            'category': 'Focus & Concentration',
            'icon': '🎯',
            'level': 'LOW',
            'badge_color': '#FEF2F2',
            'border_color': '#FECACA',
            'text_color': '#991B1B',
            'title': 'Attention Drop Detected',
            'insight': f"Average attention was {avg_focus:.1f}%, with {distracted_pct:.0f}% distracted time.",
            'actionable_advice': "Break down your study tasks into smaller 15-minute goals and remove nearby distractions like phones."
        })

    # fatigue and drowsiness
    if drowsy_episodes >= 2 or drowsy_pct >= 15.0 or total_yawns >= 4:
        suggestions.append({
            'category': 'Fatigue & Drowsiness',
            'icon': '😴',
            'level': 'HIGH_FATIGUE',
            'badge_color': '#FEF2F2',
            'border_color': '#FECACA',
            'text_color': '#991B1B',
            'title': 'Fatigue Detected',
            'insight': f"Detected {drowsy_episodes} prolonged eye-closures and {total_yawns} yawns ({drowsy_pct:.0f}% of session).",
            'actionable_advice': "Take a 15-20 minute power nap or step outside for fresh air before continuing to study."
        })
    elif total_yawns >= 1 or drowsy_episodes == 1:
        suggestions.append({
            'category': 'Fatigue & Drowsiness',
            'icon': '🥱',
            'level': 'MILD_FATIGUE',
            'badge_color': '#FFFBEB',
            'border_color': '#FDE68A',
            'text_color': '#92400E',
            'title': 'Mild Fatigue Noticed',
            'insight': f"Logged {total_yawns} yawn(s) during the session.",
            'actionable_advice': "Drink a glass of cold water and stretch to refresh yourself."
        })
    else:
        suggestions.append({
            'category': 'Fatigue & Drowsiness',
            'icon': '⚡',
            'level': 'OPTIMAL',
            'badge_color': '#ECFDF5',
            'border_color': '#A7F3D0',
            'text_color': '#065F46',
            'title': 'Good Energy Level',
            'insight': "No prolonged drowsiness or frequent yawning detected.",
            'actionable_advice': "Energy and alertness were stable throughout this study session."
        })

    # blink rate & eye strain
    if avg_bpm < 10.0 and duration_min >= 2.0:
        suggestions.append({
            'category': 'Eye Strain & Blinking',
            'icon': '👀',
            'level': 'WARNING',
            'badge_color': '#FFFBEB',
            'border_color': '#FDE68A',
            'text_color': '#92400E',
            'title': 'Low Blink Rate',
            'insight': f"Blink rate averaged {avg_bpm:.1f} BPM (healthy normal is 14-18 BPM).",
            'actionable_advice': "Remember the 20-20-20 rule: every 20 minutes look at something 20 feet away for 20 seconds."
        })
    elif avg_bpm > 28.0:
        suggestions.append({
            'category': 'Eye Strain & Blinking',
            'icon': '👀',
            'level': 'WARNING',
            'badge_color': '#FFFBEB',
            'border_color': '#FDE68A',
            'text_color': '#92400E',
            'title': 'High Blink Rate',
            'insight': f"Recorded an elevated blink rate of {avg_bpm:.1f} BPM.",
            'actionable_advice': "Check for monitor glare or increase text size to reduce eye strain."
        })
    else:
        suggestions.append({
            'category': 'Eye Strain & Blinking',
            'icon': '👀',
            'level': 'HEALTHY',
            'badge_color': '#ECFDF5',
            'border_color': '#A7F3D0',
            'text_color': '#065F46',
            'title': 'Normal Blink Rate',
            'insight': f"Blink rate was {avg_bpm:.1f} BPM (Total: {total_blinks} blinks).",
            'actionable_advice': "Normal blinking rhythm maintained."
        })

    # posture & head orientation
    if distracted_pct >= 25.0:
        suggestions.append({
            'category': 'Posture & Orientation',
            'icon': '🧘',
            'level': 'POSTURE_ALERT',
            'badge_color': '#EFF6FF',
            'border_color': '#BFDBFE',
            'text_color': '#1E40AF',
            'title': 'Frequent Head Tilts / Off-Screen Look',
            'insight': f"Head was turned or tilted away for {distracted_pct:.0f}% of the time.",
            'actionable_advice': "Elevate your study notes or books to eye level to avoid neck strain."
        })
    else:
        suggestions.append({
            'category': 'Posture & Orientation',
            'icon': '🧘',
            'level': 'GOOD_POSTURE',
            'badge_color': '#EFF6FF',
            'border_color': '#BFDBFE',
            'text_color': '#1E40AF',
            'title': 'Centered Screen Alignment',
            'insight': "Head position stayed centered towards the screen.",
            'actionable_advice': "Maintain your current screen height and chair position."
        })

    # immediate next steps
    if duration_min >= 40.0 or drowsy_episodes >= 2:
        next_step = "Take a 10-15 minute break away from screens."
    elif duration_min >= 20.0:
        next_step = "Take a quick 3-5 minute water and stretch break."
    else:
        next_step = "Ready to continue. Plan for a 25-minute Pomodoro study sprint."

    suggestions.append({
        'category': 'Recommended Next Action',
        'icon': '📋',
        'level': 'NEXT_STEPS',
        'badge_color': '#F8FAFC',
        'border_color': '#CBD5E1',
        'text_color': '#0F172A',
        'title': 'Next Step',
        'insight': f"Based on {duration_min:.1f} minutes of study.",
        'actionable_advice': next_step
    })

    return suggestions


# initialize session variables
def init_session_state():
    if 'detector' not in st.session_state:
        st.session_state.detector = FaceMeshDetector()
    if 'blink_tracker' not in st.session_state or not hasattr(st.session_state.blink_tracker, '_warmup_frames'):
        st.session_state.blink_tracker = BlinkTracker()
    if 'yawn_tracker' not in st.session_state:
        st.session_state.yawn_tracker = YawnTracker()
    if 'sleep_tracker' not in st.session_state:
        st.session_state.sleep_tracker = SleepTracker(timeout_seconds=DEFAULT_SLEEP_TIMEOUT_SECONDS)
    if 'scorer' not in st.session_state:
        st.session_state.scorer = AttentionScorer()
    if 'classifier' not in st.session_state:
        st.session_state.classifier = AttentionNeuralClassifier()
    if 'recommender' not in st.session_state or not hasattr(st.session_state.recommender, 'generate_session_suggestions'):
        st.session_state.recommender = BreakRecommender()
    if 'alert_manager' not in st.session_state or not hasattr(st.session_state.alert_manager, 'start_sleep_alarm'):
        st.session_state.alert_manager = AlertManager(enabled=True)
    if 'logger' not in st.session_state:
        st.session_state.logger = SessionLogger()
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False


init_session_state()

# sidebar settings
st.sidebar.title("⚙️ FocusGuard Settings")

input_mode = st.sidebar.radio(
    "Video Source",
    ["Live Webcam", "Phone Webcam (DroidCam URL)", "Upload Video File"],
    index=0
)

camera_index = 0
uploaded_video_file = None
stream_url = ""

if input_mode == "Live Webcam":
    camera_index = st.sidebar.number_input(
        "Camera Index (Change to 1 or 2 if seeing DroidCam)",
        min_value=0, max_value=5, value=1, step=1
    )
    st.sidebar.caption(
        "Tip: If you see DroidCam or virtual camera, change Camera Index from 0 to 1 or 2."
    )
elif input_mode == "Phone Webcam (DroidCam URL)":
    stream_url = st.sidebar.text_input("DroidCam Wi-Fi URL", value="http://192.168.1.5:4747/video")
    st.sidebar.caption("Enter phone DroidCam URL (e.g., http://192.168.x.x:4747/video).")
else:
    uploaded_video_file = st.sidebar.file_uploader("Upload Video (mp4, avi)", type=["mp4", "avi", "mov"])

st.sidebar.markdown("---")
st.sidebar.subheader("Sensitivity & Alarms")
ear_threshold = st.sidebar.slider("Eye Closure Threshold (EAR)", min_value=0.15, max_value=0.30, value=DEFAULT_EAR_THRESHOLD, step=0.01)
mar_threshold = st.sidebar.slider("Yawn Threshold (MAR)", min_value=0.40, max_value=0.80, value=DEFAULT_MAR_THRESHOLD, step=0.02)
drowsy_frames = st.sidebar.slider("Drowsiness Trigger Frames", min_value=15, max_value=50, value=DEFAULT_EAR_FRAMES_DROWSY, step=1)

# sleep alarm timeout setting
sleep_timeout = st.sidebar.slider(
    "Sleep Alarm Trigger Time (Seconds)",
    min_value=5,
    max_value=180,
    value=int(DEFAULT_SLEEP_TIMEOUT_SECONDS),
    step=5
)
st.sidebar.caption("Triggers continuous loud alarm if eyes are closed or head is down continuously. Default: 120s (2 min). Set to 5s-10s to test immediately.")

# update tracker thresholds
st.session_state.blink_tracker.ear_threshold = ear_threshold
st.session_state.blink_tracker.drowsy_frames = drowsy_frames
st.session_state.yawn_tracker.mar_threshold = mar_threshold
st.session_state.sleep_tracker.timeout_seconds = sleep_timeout

enable_audio = st.sidebar.checkbox("🔊 Audio Alerts & Sleep Alarm", value=True)
st.session_state.alert_manager.set_enabled(enable_audio)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Study Session"):
    st.session_state.scorer.reset()
    st.session_state.blink_tracker.reset()
    st.session_state.yawn_tracker.reset()
    st.session_state.sleep_tracker.reset()
    st.session_state.alert_manager.stop_sleep_alarm()
    st.session_state.logger.clear()
    st.sidebar.success("Session reset.")

# main header
st.markdown('<div class="main-header">🎯 FocusGuard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Student Attention and Fatigue Monitoring System</div>',
    unsafe_allow_html=True
)

# sleep alarm banner and manual stop button
alarm_banner_slot = st.empty()
if st.session_state.sleep_tracker.alarm_triggered or st.session_state.alert_manager.is_sleep_alarm_active():
    with alarm_banner_slot.container():
        st.markdown(f"""
        <div style="background-color: #FEF2F2; border: 3px solid #DC2626; border-radius: 12px; padding: 18px 24px; text-align: center; margin-bottom: 14px; box-shadow: 0 4px 14px rgba(220, 38, 38, 0.25);">
            <h2 style="color: #991B1B; margin: 0 0 6px 0; font-size: 1.8rem;">🚨 SLEEP ALARM ACTIVE — WAKE UP!</h2>
            <p style="color: #7F1D1D; font-size: 1.05rem; margin: 0 0 12px 0; font-weight: 600;">
                {st.session_state.sleep_tracker.trigger_reason} detected! Alarm will NOT stop until you manually stop it below.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔕 STOP ALARM / I'M AWAKE", type="primary", use_container_width=True, key="stop_alarm_main"):
            st.session_state.alert_manager.stop_sleep_alarm()
            st.session_state.sleep_tracker.dismiss()
            st.rerun()

# control buttons
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
with col_ctrl1:
    start_clicked = st.button("▶️ Start Monitoring", use_container_width=True, type="primary")
with col_ctrl2:
    stop_clicked = st.button("⏹️ Stop Monitoring", use_container_width=True)

if start_clicked:
    st.session_state.is_running = True
if stop_clicked:
    st.session_state.is_running = False
    st.session_state.alert_manager.stop_sleep_alarm()

# main layout columns
col_video, col_telemetry = st.columns([1.3, 1.0])

with col_video:
    st.markdown("### 📷 Video Feed")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col_telemetry:
    st.markdown("### 📊 Live Analytics")
    score_metric_placeholder = st.empty()
    status_badge_placeholder = st.empty()
    
    # 4 metric cards
    metrics_row = st.columns(2)
    with metrics_row[0]:
        m1_placeholder = st.empty()
    with metrics_row[1]:
        m2_placeholder = st.empty()
    
    metrics_row2 = st.columns(2)
    with metrics_row2[0]:
        m3_placeholder = st.empty()
    with metrics_row2[1]:
        m4_placeholder = st.empty()

    st.markdown("#### 📈 Attention Trend")
    chart_placeholder = st.empty()

    st.markdown("#### 💡 Smart Break & Sleep Advisor")
    break_advisor_placeholder = st.empty()

# video processing loop
if st.session_state.is_running:
    # open selected video source
    if input_mode == "Live Webcam":
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(camera_index))
    elif input_mode == "Phone Webcam (DroidCam URL)":
        if stream_url.strip():
            cap = cv2.VideoCapture(stream_url.strip())
        else:
            st.warning("Please enter a valid DroidCam IP URL.")
            st.session_state.is_running = False
            cap = None
    else:
        if uploaded_video_file is not None:
            temp_video_path = "temp_uploaded_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video_file.read())
            cap = cv2.VideoCapture(temp_video_path)
        else:
            st.warning("Please upload a video file or switch to Live Webcam.")
            st.session_state.is_running = False
            cap = None

    if cap is not None and not cap.isOpened():
        st.error(f"Could not open video source ({input_mode}).")
        st.session_state.is_running = False

    while st.session_state.is_running and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.info("Video stream stopped.")
            break

        img_h, img_w = frame.shape[:2]
        now = time.time()

        # detect facial landmarks
        extracted = st.session_state.detector.process_frame(frame)

        if extracted is not None:
            # calculate ear and mar
            ear_l = calculate_ear(extracted['left_eye_pts'])
            ear_r = calculate_ear(extracted['right_eye_pts'])
            ear_avg = (ear_l + ear_r) / 2.0
            mar = calculate_mar(extracted['mouth_pts'])

            # head pose angles
            pitch, yaw, roll, nose_proj = estimate_head_pose(
                extracted['landmarks_pixel'], img_w, img_h
            )

            # update blink and yawn trackers
            is_blink, is_prolonged_closure, total_blinks, bpm = st.session_state.blink_tracker.update(ear_avg)
            is_yawn, total_yawns = st.session_state.yawn_tracker.update(mar)

            # check sleep conditions (eyes closed or head down)
            effective_thresh = max(0.16, min(ear_threshold, st.session_state.blink_tracker.open_eye_baseline * 0.72))
            is_eyes_closed = (ear_avg < effective_thresh)
            is_head_down = (pitch < -HEAD_PITCH_DOWN_THRESHOLD)

            sleep_alarm, sleep_reason, closed_dur, head_dur = st.session_state.sleep_tracker.update(
                is_eyes_closed=is_eyes_closed,
                is_head_down=is_head_down,
                current_time=now
            )

            # trigger sleep alarm or normal microsleep alert
            if sleep_alarm:
                st.session_state.alert_manager.start_sleep_alarm()
            elif is_prolonged_closure:
                st.session_state.alert_manager.trigger_alert("drowsy")

            # neural classification
            closure_ratio = st.session_state.blink_tracker.closed_frame_count / float(drowsy_frames)
            yawn_ratio = st.session_state.yawn_tracker.open_mouth_frames / float(DEFAULT_MAR_FRAMES_YAWN)
            neural_state, confidence, prob_dict = st.session_state.classifier.predict(
                ear_l, ear_r, ear_avg, mar,
                yaw, pitch, roll, bpm,
                closure_ratio, yawn_ratio
            )

            # update attention score
            attention_score, final_state = st.session_state.scorer.update(
                has_face=True,
                ear_avg=ear_avg,
                is_prolonged_closure=is_prolonged_closure,
                mar=mar,
                is_yawning=is_yawn,
                yaw=yaw,
                pitch=pitch,
                bpm=bpm,
                ear_thresh=ear_threshold,
                neural_state=neural_state
            )

            # draw annotations
            annotated_frame = st.session_state.detector.draw_annotations(
                frame, extracted, pitch, yaw, roll, nose_proj,
                final_state, ear_avg, mar, attention_score
            )

            # overlay sleep alarm banner on video if ringing
            if st.session_state.alert_manager.is_sleep_alarm_active():
                cv2.rectangle(annotated_frame, (0, 0), (img_w, 60), (0, 0, 220), -1)
                cv2.putText(
                    annotated_frame, f"SLEEP ALARM! WAKE UP! ({sleep_reason})",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
                )

            # log entry
            st.session_state.logger.log_entry(
                now - st.session_state.scorer.start_time,
                ear_avg, mar, yaw, pitch, bpm, attention_score, final_state
            )

        else:
            # no face detected
            # if head was already tilting down when face was lost, continue tracking head down
            was_head_down = (st.session_state.sleep_tracker.head_down_start is not None)
            sleep_alarm, sleep_reason, closed_dur, head_dur = st.session_state.sleep_tracker.update(
                is_eyes_closed=False,
                is_head_down=was_head_down,
                current_time=now
            )
            if sleep_alarm:
                st.session_state.alert_manager.start_sleep_alarm()

            attention_score, final_state = st.session_state.scorer.update(
                has_face=False,
                ear_avg=0.0,
                is_prolonged_closure=False,
                mar=0.0,
                is_yawning=False,
                yaw=0.0,
                pitch=0.0,
                bpm=0.0,
                ear_thresh=ear_threshold
            )
            annotated_frame = frame.copy()
            cv2.putText(
                annotated_frame, "NO FACE DETECTED - PLEASE FACE CAMERA",
                (30, int(img_h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
            )
            if st.session_state.alert_manager.is_sleep_alarm_active():
                cv2.rectangle(annotated_frame, (0, 0), (img_w, 60), (0, 0, 220), -1)
                cv2.putText(
                    annotated_frame, "SLEEP ALARM ACTIVE! HEAD DOWN DETECTED!",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
                )

            ear_avg = 0.0
            mar = 0.0
            bpm = 0.0
            total_blinks = st.session_state.blink_tracker.total_blinks
            total_yawns = st.session_state.yawn_tracker.total_yawns

        # display frame in streamlit
        rgb_disp = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_disp, channels="RGB", use_container_width=True)

        # live telemetry
        summary = st.session_state.scorer.get_summary()

        score_metric_placeholder.metric(
            label="Focus Index",
            value=f"{attention_score:.1f}%",
            delta=f"{attention_score - summary['avg_attention']:.1f}% vs Avg"
        )

        badge_class = {
            "Focused": "status-badge-focused",
            "Distracted": "status-badge-distracted",
            "Drowsy": "status-badge-drowsy",
            "Away": "status-badge-away"
        }.get(final_state, "status-badge-away")

        status_badge_placeholder.markdown(
            f'<div class="{badge_class}">Current State: {final_state.upper()}</div>',
            unsafe_allow_html=True
        )

        # metric cards
        m1_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{bpm:.0f}</div>
            <div class="metric-label">Blinks / Min</div>
        </div>
        """, unsafe_allow_html=True)

        m2_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_yawns}</div>
            <div class="metric-label">Yawns Detected</div>
        </div>
        """, unsafe_allow_html=True)

        m3_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['session_duration_min']:.1f} m</div>
            <div class="metric-label">Study Duration</div>
        </div>
        """, unsafe_allow_html=True)

        m4_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['focused_pct']:.0f}%</div>
            <div class="metric-label">Focused Time</div>
        </div>
        """, unsafe_allow_html=True)

        # live line chart
        if len(st.session_state.scorer.history_scores) > 1:
            recent_df = pd.DataFrame({
                'Time (s)': st.session_state.scorer.history_timestamps[-40:],
                'Attention (%)': st.session_state.scorer.history_scores[-40:]
            }).set_index('Time (s)')
            chart_placeholder.line_chart(recent_df, height=180)

        # break & sleep recommendations during session
        if st.session_state.alert_manager.is_sleep_alarm_active():
            break_advisor_placeholder.markdown(f"""
            <div style="background-color: #FDE8E8; padding: 14px; border-radius: 10px; border: 2px solid #DC2626; color: #991B1B;">
                <b style="font-size: 1.0rem;">🚨 SLEEP ALARM ACTIVE</b><br>
                <span style="font-size: 0.9rem;">{st.session_state.sleep_tracker.trigger_reason}! Click 'Stop Alarm' above to dismiss.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            break_rec = st.session_state.recommender.evaluate(
                session_duration_sec=summary['session_duration_sec'],
                total_yawns=total_yawns,
                drowsy_episodes=st.session_state.blink_tracker.drowsy_episode_count,
                low_attention_streak_sec=st.session_state.scorer.low_attention_streak_seconds,
                current_attention=attention_score,
                avg_attention=summary['avg_attention']
            )

            level_color = {
                'CRITICAL': '#FDE8E8',
                'WARNING': '#FEF08A',
                'INFO': '#E1EFFE',
                'GOOD': '#DEF7EC',
                'NORMAL': '#F3F4F6'
            }.get(break_rec['level'], '#F3F4F6')

            sleep_guard_info = ""
            if closed_dur > 1.0:
                sleep_guard_info = f" | Eyes closed: {int(closed_dur)}s / {int(sleep_timeout)}s"
            elif head_dur > 1.0:
                sleep_guard_info = f" | Head down: {int(head_dur)}s / {int(sleep_timeout)}s"

            break_advisor_placeholder.markdown(f"""
            <div style="background-color: {level_color}; padding: 14px; border-radius: 10px; border: 1px solid #CBD5E1; color: #1E293B; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <b style="color: #0F172A; font-size: 0.95rem;">{break_rec['badge']}{sleep_guard_info}</b><br>
                <span style="font-size: 0.9rem; color: #334155; line-height: 1.4;">{break_rec['message']}</span>
            </div>
            """, unsafe_allow_html=True)

        time.sleep(0.03)

    if cap is not None:
        cap.release()

else:
    # standby view
    video_placeholder.info("Click 'Start Monitoring' above to activate webcam tracking.")
    
    # show summary after session
    if len(st.session_state.scorer.history_scores) > 0:
        st.markdown("### 📋 Completed Session Summary")
        summary = st.session_state.scorer.get_summary()

        sum_cols = st.columns(4)
        sum_cols[0].metric("Average Focus", f"{summary['avg_attention']}%")
        sum_cols[1].metric("Session Duration", f"{summary['session_duration_min']} min")
        sum_cols[2].metric("Total Blinks", st.session_state.blink_tracker.total_blinks)
        sum_cols[3].metric("Total Yawns", st.session_state.yawn_tracker.total_yawns)

        # export csv
        csv_path = st.session_state.logger.export_csv()
        df_log = st.session_state.logger.get_dataframe()
        csv_bytes = df_log.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Study Session Report (CSV)",
            data=csv_bytes,
            file_name="focusguard_study_session.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("### 💡 Study Insights & Recommendations")
        st.caption("Feedback based on your attention trends, blink rate, and fatigue metrics:")

        # calculate average bpm
        total_time_min = max(0.1, summary['session_duration_min'])
        avg_bpm = round(st.session_state.blink_tracker.total_blinks / total_time_min, 1)

        # generate suggestions
        suggestions = get_personalized_suggestions(
            summary=summary,
            total_blinks=st.session_state.blink_tracker.total_blinks,
            avg_bpm=avg_bpm,
            total_yawns=st.session_state.yawn_tracker.total_yawns,
            drowsy_episodes=st.session_state.blink_tracker.drowsy_episode_count
        )

        for item in suggestions:
            st.markdown(f"""
            <div style="background-color: {item['badge_color']}; border: 1px solid {item['border_color']}; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 1.0rem; color: {item['text_color']};">
                        {item['icon']} {item['category']} — {item['title']}
                    </span>
                    <span style="font-size: 0.75rem; font-weight: 700; background: rgba(255,255,255,0.85); color: {item['text_color']}; padding: 3px 10px; border-radius: 12px; border: 1px solid {item['border_color']};">
                        {item['level']}
                    </span>
                </div>
                <div style="font-size: 0.90rem; color: #334155; margin-bottom: 6px; line-height: 1.4;">
                    <b>Report Insight:</b> {item['insight']}
                </div>
                <div style="font-size: 0.90rem; color: #0F172A; line-height: 1.4;">
                    <b>Actionable Advice:</b> {item['actionable_advice']}
                </div>
            </div>
            """, unsafe_allow_html=True)
