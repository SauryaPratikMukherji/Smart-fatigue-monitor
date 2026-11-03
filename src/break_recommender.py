import time
from typing import Dict, Any
from src.config import (
    MAX_CONTINUOUS_STUDY_MINUTES,
    DROWSY_EPISODES_BEFORE_BREAK,
    YAWNS_BEFORE_BREAK,
    LOW_ATTENTION_DURATION_BREAK
)


# study break recommender
class BreakRecommender:
    def __init__(self):
        self.last_break_prompt_time = 0.0

    # check current metrics and return live break recommendation
    def evaluate(
        self,
        session_duration_sec: float,
        total_yawns: int,
        drowsy_episodes: int,
        low_attention_streak_sec: float,
        current_attention: float,
        avg_attention: float
    ) -> Dict[str, Any]:
        session_minutes = session_duration_sec / 60.0

        # microsleep or heavy fatigue
        if drowsy_episodes >= DROWSY_EPISODES_BEFORE_BREAK:
            return {
                'level': 'CRITICAL',
                'badge': '🛑 Rest Needed',
                'title': 'High Fatigue / Microsleep Detected',
                'message': (
                    f"You have had {drowsy_episodes} prolonged eye closures. "
                    "You seem very tired. Please take a 15-20 minute rest or walking break."
                ),
                'action': 'Take 15-20 min Break'
            }

        # multiple yawns or prolonged low focus
        if total_yawns >= YAWNS_BEFORE_BREAK or low_attention_streak_sec >= LOW_ATTENTION_DURATION_BREAK:
            return {
                'level': 'WARNING',
                'badge': '⚠️ Break Recommended',
                'title': 'Attention Drop & Fatigue Pattern',
                'message': (
                    f"Detected {total_yawns} yawns and attention has been low for "
                    f"{int(low_attention_streak_sec)} seconds. Take a 5-minute break to drink water and stretch."
                ),
                'action': 'Take 5 min Break'
            }

        # long study duration
        if session_minutes >= MAX_CONTINUOUS_STUDY_MINUTES:
            return {
                'level': 'WARNING',
                'badge': '⏰ Study Stint Completed',
                'title': f'{int(session_minutes)} Minutes Continuous Study',
                'message': (
                    f"You have been studying continuously for {int(session_minutes)} minutes. "
                    "Take a 10-minute break away from the screen."
                ),
                'action': 'Take 10 min Rest'
            }

        # 20 minute eye rest check
        if 20.0 <= session_minutes < 25.0:
            return {
                'level': 'INFO',
                'badge': '👀 20-20-20 Eye Rest',
                'title': 'Rest Your Eyes',
                'message': (
                    "Look at an object 20 feet away for 20 seconds to relax your eyes."
                ),
                'action': '20s Eye Rest'
            }

        # good focus
        if current_attention >= 70.0:
            return {
                'level': 'GOOD',
                'badge': '✨ Good Focus',
                'title': 'Concentration Active',
                'message': (
                    "Good job! Your gaze is steady and attention is strong. Keep studying."
                ),
                'action': 'Keep Studying'
            }

        return {
            'level': 'NORMAL',
            'badge': '📘 Study In Progress',
            'title': 'Session Active',
            'message': (
                "Maintain good posture and keep your screen at eye level."
            ),
            'action': 'Maintain Posture'
        }

    # generate recommendations based on final session numbers
    def generate_session_suggestions(
        self,
        summary: Dict[str, Any],
        total_blinks: int,
        avg_bpm: float,
        total_yawns: int,
        drowsy_episodes: int
    ) -> list:
        suggestions = []
        duration_min = summary.get('session_duration_min', 0.0)
        avg_focus = summary.get('avg_attention', 0.0)
        focused_pct = summary.get('focused_pct', 0.0)
        distracted_pct = summary.get('distracted_pct', 0.0)
        drowsy_pct = summary.get('drowsy_pct', 0.0)

        # 1. focus analysis
        if avg_focus >= 80.0 and focused_pct >= 75.0:
            suggestions.append({
                'category': 'Focus & Concentration',
                'icon': '🎯',
                'level': 'GOOD',
                'badge_color': '#ECFDF5',
                'border_color': '#A7F3D0',
                'text_color': '#065F46',
                'title': 'High Focus Session',
                'insight': f"Average attention was {avg_focus:.1f}% with {focused_pct:.0f}% of time spent focused.",
                'actionable_advice': "Keep study sessions between 45 to 50 minutes followed by a short 5-minute break to avoid getting tired."
            })
        elif avg_focus >= 60.0:
            suggestions.append({
                'category': 'Focus & Concentration',
                'icon': '🎯',
                'level': 'MODERATE',
                'badge_color': '#FFFBEB',
                'border_color': '#FDE68A',
                'text_color': '#92400E',
                'title': 'Moderate Focus with Some Distractions',
                'insight': f"Average attention was {avg_focus:.1f}%, with {distracted_pct:.0f}% of the time looking away.",
                'actionable_advice': "Try studying in 25-minute blocks and keep your phone in another room to reduce glancing away."
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
                'insight': f"Average attention was {avg_focus:.1f}%, and focused time was only {focused_pct:.0f}%.",
                'actionable_advice': "Break down your study topic into smaller 15-minute goals and study in a quiet spot."
            })

        # 2. fatigue and yawns
        if drowsy_episodes >= 2 or drowsy_pct >= 15.0 or total_yawns >= 4:
            suggestions.append({
                'category': 'Fatigue & Sleepiness',
                'icon': '😴',
                'level': 'HIGH',
                'badge_color': '#FEF2F2',
                'border_color': '#FECACA',
                'text_color': '#991B1B',
                'title': 'High Fatigue Detected',
                'insight': f"Detected {drowsy_episodes} eye closure events and {total_yawns} yawns ({drowsy_pct:.0f}% of the time).",
                'actionable_advice': "Take a 15-20 minute power nap or walk around. Avoid studying in bed or dim light."
            })
        elif total_yawns >= 1 or drowsy_episodes == 1:
            suggestions.append({
                'category': 'Fatigue & Sleepiness',
                'icon': '🥱',
                'level': 'MILD',
                'badge_color': '#FFFBEB',
                'border_color': '#FDE68A',
                'text_color': '#92400E',
                'title': 'Mild Fatigue',
                'insight': f"Recorded {total_yawns} yawn(s) and slight eye droop.",
                'actionable_advice': "Drink a glass of cold water and stretch for 1-2 minutes to stay alert."
            })
        else:
            suggestions.append({
                'category': 'Fatigue & Sleepiness',
                'icon': '⚡',
                'level': 'GOOD',
                'badge_color': '#ECFDF5',
                'border_color': '#A7F3D0',
                'text_color': '#065F46',
                'title': 'Good Energy Level',
                'insight': "No signs of drowsiness or frequent yawning detected during this session.",
                'actionable_advice': "Good alertness throughout. Keep up your regular sleep and hydration routine."
            })

        # 3. blink rate check
        if avg_bpm < 10.0 and duration_min >= 2.0:
            suggestions.append({
                'category': 'Eye Strain & Blinking',
                'icon': '👀',
                'level': 'WARNING',
                'badge_color': '#FFFBEB',
                'border_color': '#FDE68A',
                'text_color': '#92400E',
                'title': 'Low Blink Rate (Screen Staring)',
                'insight': f"Average blink rate was {avg_bpm:.1f} BPM, which is lower than normal (14-18 BPM).",
                'actionable_advice': "Remember to blink more often when reading on screen and look away every 20 minutes to prevent dry eyes."
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
                'insight': f"Blink rate was high ({avg_bpm:.1f} BPM), which can happen from screen glare or eye fatigue.",
                'actionable_advice': "Lower your screen brightness to match your room light and avoid strong reflections on your screen."
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
                'insight': f"Average blink frequency was {avg_bpm:.1f} BPM (Total: {total_blinks} blinks).",
                'actionable_advice': "Blink rate remained normal and healthy during study."
            })

        # 4. head posture check
        if distracted_pct >= 25.0:
            suggestions.append({
                'category': 'Desk Posture',
                'icon': '🧘',
                'level': 'POSTURE',
                'badge_color': '#EFF6FF',
                'border_color': '#BFDBFE',
                'text_color': '#1E40AF',
                'title': 'Frequent Head Tilting',
                'insight': f"Turned head or looked down for {distracted_pct:.0f}% of the time.",
                'actionable_advice': "If looking down at notes or phone, elevate them with a book stand to keep your neck straight."
            })
        else:
            suggestions.append({
                'category': 'Desk Posture',
                'icon': '🧘',
                'level': 'GOOD',
                'badge_color': '#EFF6FF',
                'border_color': '#BFDBFE',
                'text_color': '#1E40AF',
                'title': 'Good Head Posture',
                'insight': "Head remained upright and facing the screen most of the session.",
                'actionable_advice': "Maintain this monitor height and sit comfortably upright."
            })

        # 5. next action suggestion
        if duration_min >= 40.0 or drowsy_episodes >= 2:
            next_step = "Take a 10-15 minute break away from screens before starting your next session."
        elif duration_min >= 20.0:
            next_step = "Take a quick 3-5 minute water break before continuing."
        else:
            next_step = "You are in a good rhythm. Continue for another 20-25 minutes."

        suggestions.append({
            'category': 'Next Step',
            'icon': '📋',
            'level': 'ACTION',
            'badge_color': '#F8FAFC',
            'border_color': '#CBD5E1',
            'text_color': '#0F172A',
            'title': 'Recommended Action',
            'insight': f"Based on your {duration_min:.1f} minute study time.",
            'actionable_advice': next_step
        })

        return suggestions
