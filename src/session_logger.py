# session logger and csv exporter
import os
import csv
import time
import pandas as pd
from typing import List, Dict, Any


class SessionLogger:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    # log a single time point
    def log_entry(
        self,
        timestamp: float,
        ear: float,
        mar: float,
        yaw: float,
        pitch: float,
        bpm: float,
        attention_score: float,
        state: str
    ):
        self.records.append({
            'elapsed_sec': round(timestamp, 1),
            'ear': round(ear, 3),
            'mar': round(mar, 3),
            'yaw': round(yaw, 1),
            'pitch': round(pitch, 1),
            'blink_bpm': round(bpm, 1),
            'attention_score': round(attention_score, 1),
            'state': state
        })

    # save logged data to csv file
    def export_csv(self, filename: str = None) -> str:
        if not filename:
            timestr = time.strftime("%Y%m%d_%H%M%S")
            filename = f"focusguard_session_{timestr}.csv"

        filepath = os.path.join(self.output_dir, filename)

        # export dataframe to csv
        if not self.records:
            df = pd.DataFrame(columns=[
                'elapsed_sec', 'ear', 'mar', 'yaw', 'pitch',
                'blink_bpm', 'attention_score', 'state'
            ])
        else:
            df = pd.DataFrame(self.records)

        df.to_csv(filepath, index=False)
        return filepath

    # get records as dataframe
    def get_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=[
                'elapsed_sec', 'ear', 'mar', 'yaw', 'pitch',
                'blink_bpm', 'attention_score', 'state'
            ])
        return pd.DataFrame(self.records)

    # reset records list
    def clear(self):
        self.records.clear()
