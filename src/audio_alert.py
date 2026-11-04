# audio alert manager with sound alerts and sleep alarm
import sys
import threading
import time

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlertManager:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._is_beeping = False
        self._last_alert_time = 0.0
        self._cooldown_seconds = 2.0
        self._sleep_alarm_active = False
        self._sleep_alarm_thread = None

    # play a brief warning sound for microsleep or break
    def trigger_alert(self, alert_type: str = "drowsy"):
        if not self.enabled or self._sleep_alarm_active:
            return

        current_time = time.time()
        if current_time - self._last_alert_time < self._cooldown_seconds:
            return

        if self._is_beeping:
            return

        self._last_alert_time = current_time
        threading.Thread(target=self._play_tone, args=(alert_type,), daemon=True).start()

    def _play_tone(self, alert_type: str):
        self._is_beeping = True
        try:
            if HAS_WINSOUND and sys.platform == "win32":
                if alert_type == "drowsy":
                    winsound.Beep(1800, 250)
                    time.sleep(0.08)
                    winsound.Beep(1800, 350)
                elif alert_type == "break":
                    winsound.Beep(880, 150)
                    time.sleep(0.05)
                    winsound.Beep(1100, 200)
                else:
                    winsound.Beep(1000, 200)
            else:
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            self._is_beeping = False

    # start continuous loud sleep alarm
    def start_sleep_alarm(self):
        if not self.enabled:
            return
        if self._sleep_alarm_active:
            return

        self._sleep_alarm_active = True
        self._sleep_alarm_thread = threading.Thread(
            target=self._run_sleep_alarm_loop, daemon=True
        )
        self._sleep_alarm_thread.start()

    # stop continuous sleep alarm manually
    def stop_sleep_alarm(self):
        self._sleep_alarm_active = False

    # check if alarm is currently ringing
    def is_sleep_alarm_active(self) -> bool:
        return self._sleep_alarm_active

    # background loop for continuous loud ringing
    def _run_sleep_alarm_loop(self):
        while self._sleep_alarm_active:
            try:
                if HAS_WINSOUND and sys.platform == "win32":
                    # two-tone siren
                    winsound.Beep(2400, 220)
                    if not self._sleep_alarm_active:
                        break
                    time.sleep(0.04)
                    winsound.Beep(1700, 220)
                    if not self._sleep_alarm_active:
                        break
                    time.sleep(0.08)
                else:
                    sys.stdout.write('\a')
                    sys.stdout.flush()
                    time.sleep(0.4)
            except Exception:
                time.sleep(0.2)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.stop_sleep_alarm()
