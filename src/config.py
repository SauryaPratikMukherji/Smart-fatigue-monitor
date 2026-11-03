# landmark indices and project settings

# eye landmark indices for EAR
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# eye contour indices for drawing lines
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# mouth landmarks for MAR
MOUTH_INNER_LIPS = {
    'top': 13,
    'bottom': 14,
    'left_corner': 308,
    'right_corner': 78,
    'top_left': 312,
    'bottom_left': 317,
    'top_right': 82,
    'bottom_right': 87
}

# 3D points for head pose estimation
POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

MODEL_3D_POINTS = [
    (0.0, 0.0, 0.0),             # nose tip
    (0.0, -330.0, -65.0),         # chin
    (-225.0, 170.0, -135.0),      # left eye corner
    (225.0, 170.0, -135.0),       # right eye corner
    (-150.0, -150.0, -125.0),     # left mouth corner
    (150.0, -150.0, -125.0)       # right mouth corner
]

# thresholds
DEFAULT_EAR_THRESHOLD = 0.24          # eye closure threshold
DEFAULT_EAR_FRAMES_DROWSY = 25        # frames closed to count as drowsy
DEFAULT_EAR_FRAMES_BLINK = 1          # frames to count as blink

DEFAULT_MAR_THRESHOLD = 0.58          # mouth open threshold
DEFAULT_MAR_FRAMES_YAWN = 25          # frames open to count as yawn

# head pose angles in degrees
HEAD_YAW_THRESHOLD = 25.0             # looking left or right
HEAD_PITCH_DOWN_THRESHOLD = 20.0      # looking down
HEAD_PITCH_UP_THRESHOLD = 22.0        # looking up

# rolling window settings
BLINK_WINDOW_SECONDS = 60             # calculate blinks per minute over 60s
SMOOTHING_ALPHA = 0.15                # smoothing factor for score

# break trigger settings
MAX_CONTINUOUS_STUDY_MINUTES = 45
DROWSY_EPISODES_BEFORE_BREAK = 3
YAWNS_BEFORE_BREAK = 4
LOW_ATTENTION_DURATION_BREAK = 90
