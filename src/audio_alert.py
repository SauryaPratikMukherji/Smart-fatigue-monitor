import sys
import threading
import time

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlertManager:
    # plays sound alerts in background thread so video does not freeze
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._is_beeping = False
        self._last_alert_time = 0.0
        self._cooldown_seconds = 2.0

    def trigger_alert(self, alert_type: str = "drowsy"):
        if not self.enabled:
            return

        current_time = time.time()
        if current_time - self._last_alert_time < self._cooldown_seconds:
            return

        if self._is_beeping:
            return

        self._last_alert_time = current_time
        # start beep in new thread
        threading.Thread(target=self._play_tone, args=(alert_type,), daemon=True).start()

    def _play_tone(self, alert_type: str):
        self._is_beeping = True
        try:
            if HAS_WINSOUND and sys.platform == "win32":
                if alert_type == "drowsy":
                    # warning beep for drowsiness
                    winsound.Beep(1800, 250)
                    time.sleep(0.08)
                    winsound.Beep(1800, 350)
                elif alert_type == "break":
                    # break alert chime
                    winsound.Beep(880, 150)
                    time.sleep(0.05)
                    winsound.Beep(1100, 200)
                else:
                    winsound.Beep(1000, 200)
            else:
                # terminal beep fallback
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            self._is_beeping = False

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
