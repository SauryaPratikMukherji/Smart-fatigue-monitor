import time
from typing import Dict, List, Tuple, Optional
from src.config import (
    HEAD_YAW_THRESHOLD,
    HEAD_PITCH_DOWN_THRESHOLD,
    HEAD_PITCH_UP_THRESHOLD,
    SMOOTHING_ALPHA
)


# calculates attention score and tracks study time
class AttentionScorer:
    def __init__(self, smoothing_alpha: float = SMOOTHING_ALPHA):
        self.alpha = smoothing_alpha
        self.current_score = 100.0
        self.start_time = time.time()
        self.last_update_time = time.time()

        # history lists for graphs
        self.history_timestamps: List[float] = []
        self.history_scores: List[float] = []
        self.history_states: List[str] = []

        # time tracking
        self.time_focused = 0.0
        self.time_distracted = 0.0
        self.time_drowsy = 0.0
        self.time_away = 0.0

        self.low_attention_streak_seconds = 0.0

    # calculate raw score from metrics
    def compute_instantaneous_score(
        self,
        has_face: bool,
        ear_avg: float,
        is_prolonged_closure: bool,
        mar: float,
        is_yawning: bool,
        yaw: float,
        pitch: float,
        bpm: float,
        ear_thresh: float
    ) -> Tuple[float, str]:
        if not has_face:
            return 0.0, "Away"

        score = 100.0
        state = "Focused"

        # 1. eye closure penalty
        if is_prolonged_closure:
            score -= 65.0
            state = "Drowsy"
        elif ear_avg < ear_thresh:
            score -= 30.0

        # 2. yawn penalty
        if is_yawning:
            score -= 40.0
            if state != "Drowsy":
                state = "Drowsy"
        elif mar > 0.45:
            score -= 15.0

        # 3. head pose penalty
        abs_yaw = abs(yaw)
        if abs_yaw > HEAD_YAW_THRESHOLD:
            yaw_penalty = min(40.0, (abs_yaw - HEAD_YAW_THRESHOLD) * 1.5)
            score -= yaw_penalty
            if state == "Focused":
                state = "Distracted"

        if pitch < -HEAD_PITCH_DOWN_THRESHOLD:
            pitch_penalty = min(35.0, (abs(pitch) - HEAD_PITCH_DOWN_THRESHOLD) * 1.5)
            score -= pitch_penalty
            if state == "Focused":
                state = "Distracted"
        elif pitch > HEAD_PITCH_UP_THRESHOLD:
            score -= 20.0
            if state == "Focused":
                state = "Distracted"

        # 4. blink rate penalty
        if bpm > 35.0:
            score -= 10.0
        elif bpm < 5.0 and (time.time() - self.start_time > 30):
            score -= 10.0

        score = max(0.0, min(100.0, score))
        return score, state

    # smooth score and update time breakdown
    def update(
        self,
        has_face: bool,
        ear_avg: float,
        is_prolonged_closure: bool,
        mar: float,
        is_yawning: bool,
        yaw: float,
        pitch: float,
        bpm: float,
        ear_thresh: float,
        neural_state: Optional[str] = None
    ) -> Tuple[float, str]:
        now = time.time()
        dt = max(0.001, now - self.last_update_time)
        self.last_update_time = now

        raw_score, heuristic_state = self.compute_instantaneous_score(
            has_face, ear_avg, is_prolonged_closure, mar, is_yawning,
            yaw, pitch, bpm, ear_thresh
        )

        # decide current state
        if not has_face:
            final_state = "Away"
        elif is_prolonged_closure or is_yawning:
            final_state = "Drowsy"
        elif neural_state is not None:
            final_state = neural_state
        else:
            final_state = heuristic_state

        # exponential smoothing
        self.current_score = (self.alpha * raw_score) + ((1.0 - self.alpha) * self.current_score)
        self.current_score = max(0.0, min(100.0, self.current_score))

        # time accounting
        if final_state == "Focused":
            self.time_focused += dt
        elif final_state == "Distracted":
            self.time_distracted += dt
        elif final_state == "Drowsy":
            self.time_drowsy += dt
        else:
            self.time_away += dt

        # low attention counter
        if self.current_score < 50.0:
            self.low_attention_streak_seconds += dt
        else:
            self.low_attention_streak_seconds = max(0.0, self.low_attention_streak_seconds - dt * 2)

        # save history entry every 0.5s
        if not self.history_timestamps or (now - self.history_timestamps[-1] >= 0.5):
            elapsed_sec = round(now - self.start_time, 1)
            self.history_timestamps.append(elapsed_sec)
            self.history_scores.append(round(self.current_score, 1))
            self.history_states.append(final_state)

        return round(self.current_score, 1), final_state

    # get summary numbers for report
    def get_summary(self) -> Dict[str, float]:
        total_time = max(1.0, time.time() - self.start_time)
        avg_score = (
            sum(self.history_scores) / len(self.history_scores)
            if self.history_scores else self.current_score
        )

        return {
            'session_duration_sec': round(total_time, 1),
            'session_duration_min': round(total_time / 60.0, 1),
            'avg_attention': round(avg_score, 1),
            'time_focused_sec': round(self.time_focused, 1),
            'time_distracted_sec': round(self.time_distracted, 1),
            'time_drowsy_sec': round(self.time_drowsy, 1),
            'time_away_sec': round(self.time_away, 1),
            'focused_pct': round((self.time_focused / total_time) * 100.0, 1),
            'distracted_pct': round((self.time_distracted / total_time) * 100.0, 1),
            'drowsy_pct': round((self.time_drowsy / total_time) * 100.0, 1),
        }

    # reset counters
    def reset(self):
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.current_score = 100.0
        self.history_timestamps.clear()
        self.history_scores.clear()
        self.history_states.clear()
        self.time_focused = 0.0
        self.time_distracted = 0.0
        self.time_drowsy = 0.0
        self.time_away = 0.0
        self.low_attention_streak_seconds = 0.0
