# FocusGuard: Real-Time Attention & Fatigue Monitoring System

> **College Python Mini-Project**  
> **Tech Stack:** Python 3.11, OpenCV, MediaPipe Face Mesh, NumPy, SciPy, Streamlit, Pandas.

---

## 1. Project Overview

Students frequently spend long study hours in front of screens without realizing when attention drops or fatigue sets in. Extended screen exposure without breaks leads to digital eye strain, drowsiness, and reduced retention.

**FocusGuard** is a webcam-based computer vision application designed to monitor student attentiveness and fatigue indicators in real time during self-study sessions.

### Key Features
- **Eye Closure & Drowsiness Detection:** Calculates Eye Aspect Ratio (EAR) to detect prolonged eye closure and micro-sleep events.
- **Blink Counter & Frequency (BPM):** Measures rolling Blinks Per Minute to monitor ocular strain and blank staring.
- **Yawn Detection:** Uses Mouth Aspect Ratio (MAR) to detect sustained wide mouth openings.
- **Head Pose Estimation:** Uses 3D Perspective-n-Point (`solvePnP`) to estimate Pitch, Yaw, and Roll (detecting looking down at phones or looking away).
- **Neural State Classifier:** Evaluates facial metrics into Focused, Distracted, or Drowsy states.
- **Attention Index (0-100%):** Computes a smoothed focus percentage based on gaze, eye openness, and fatigue penalties.
- **Break Recommendations:** Provides rest suggestions (Pomodoro intervals and the 20-20-20 eye strain rule).
- **Two Running Modes:**
  1. **Web Dashboard (`app.py`):** Interactive Streamlit UI with metric cards, live trend charts, sensitivity sliders, and CSV report export.
  2. **Desktop Window (`main.py`):** Fast OpenCV HUD window with real-time on-screen metrics and keyboard shortcuts.
- **Audio Alert:** Beeps when prolonged drowsiness is detected (can be toggled on/off).

---

## 2. System Architecture & Methodology

```
                      +-----------------------------------+
                      |       Webcam / Video Input        |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   MediaPipe Face Mesh (468 Pts)   |
                      +-----------------------------------+
                             /          |          \
                            /           |           \
                           v            v            v
                      +---------+  +---------+  +-----------+
                      |   EAR   |  |   MAR   |  | Head Pose |
                      | (Eyes)  |  | (Mouth) |  | (solvePnP)|
                      +---------+  +---------+  +-----------+
                           \            |            /
                            \           |           /
                             v          v          v
                      +-----------------------------------+
                      |        10-Feature Vector          |
                      +-----------------------------------+
                                        |
                         +--------------+--------------+
                         |                             |
                         v                             v
               +-------------------+         +-------------------+
               | Neural Classifier |         | Attention Scorer  |
               | (State Decision)  |         | (0 - 100% Index)  |
               +-------------------+         +-------------------+
                         \                             /
                          +-------------+-------------+
                                        |
                                        v
                      +-----------------------------------+
                      |        Feedback & Alerts          |
                      |  - Audio buzzer on micro-sleep   |
                      |  - Rest & ergonomics advice       |
                      |  - CSV study report export        |
                      +-----------------------------------+
```

---

## 3. Mathematical Formulations

### 1. Eye Aspect Ratio (EAR)
Calculates vertical eyelid distances relative to horizontal eye width:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
- $p_1, p_4$: Outer and inner eye corners.
- $p_2, p_3, p_5, p_6$: Upper and lower eyelid points.
- Eye closure is detected when $\text{EAR}$ falls below the calibrated threshold.

### 2. Mouth Aspect Ratio (MAR)
Measures mouth opening relative to mouth width:
$$\text{MAR} = \frac{\|p_{\text{top}} - p_{\text{bottom}}\| + \|p_{\text{top\_left}} - p_{\text{bottom\_left}}\| + \|p_{\text{top\_right}} - p_{\text{bottom\_right}}\|}{3 \cdot \|p_{\text{left}} - p_{\text{right}}\|}$$
- Sustained high MAR for multiple consecutive frames signals a yawn.

### 3. Head Pose Estimation
Uses `cv2.solvePnP` to map 3D model head coordinates to 2D facial landmarks:
$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} R & t \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$
Rotation matrix $R$ is decomposed into Euler angles (Pitch, Yaw, Roll).

### 4. Exponential Moving Average (EMA)
To prevent noisy frame-to-frame score jumps:
$$\text{Score}_t = \alpha \cdot \text{RawScore}_t + (1 - \alpha) \cdot \text{Score}_{t-1}$$

---

## 4. Project Directory Structure

```
c:\python mini project 1\
├── src/
│   ├── __init__.py
│   ├── config.py                 # Landmark indices, default thresholds, constants
│   ├── face_mesh_detector.py     # MediaPipe FaceMesh wrapper & drawing utilities
│   ├── fatigue_metrics.py        # EAR, MAR, solvePnP Head Pose, blink/yawn trackers
│   ├── neural_classifier.py      # Neural state classifier
│   ├── attention_scorer.py       # Focus score calculation (0-100%) and metrics
│   ├── break_recommender.py      # Study break advisor logic
│   ├── audio_alert.py            # Non-blocking audio buzzer
│   └── session_logger.py         # Session metrics recorder & CSV exporter
├── models/
│   └── attention_weights.npz     # Trained neural model weights
├── reports/                      # Exported CSV study session reports
├── app.py                        # Streamlit web dashboard
├── main.py                       # Desktop OpenCV application
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 5. How to Run

Activate your virtual environment and run either interface:

### Option 1: Streamlit Web Dashboard
Run the following command in PowerShell:
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Open `http://localhost:8501` in your browser.
- Select video source (`Live Webcam`, `Phone Webcam`, or `Upload Video File`).
- Click **Start Monitoring** to begin tracking.
- Click **Stop Monitoring** to view the session report and download the CSV.

### Option 2: Desktop OpenCV Window
Run the following command in PowerShell:
```powershell
.\venv\Scripts\python.exe main.py
```
Keyboard shortcuts in the OpenCV window:
- `q`: Quit and display final session report in the terminal.
- `r`: Reset session timers and counters.
- `m`: Mute or unmute the audio alarm.

---

## 6. Viva & Presentation Questions

| Question | Explanation |
|---|---|
| **Why use MediaPipe Face Mesh instead of Haar Cascades?** | Haar Cascades only detect 2D bounding boxes and fail under head tilts. MediaPipe predicts 468 3D landmarks in real time with high accuracy. |
| **How does the system differentiate talking from yawning?** | Talking produces fast, brief mouth openings ($<0.5$ sec). Yawning produces sustained, wide mouth opening ($\text{MAR} > 0.58$) lasting for multiple consecutive frames. |
| **How is eye strain detected?** | The system measures rolling Blinks Per Minute (BPM). Staring for too long ($<10$ BPM) or rapid fluttering ($>28$ BPM) indicates visual fatigue. |
| **How does head pose detect distraction?** | When yaw or pitch exceeds angular limits (e.g. looking down at a mobile or looking sideways), the system marks the state as Distracted. |
