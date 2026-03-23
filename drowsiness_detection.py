"""
drowsiness_detection.py
-----------------------
Core logic for:
  - Eye Aspect Ratio (EAR) computation
  - Mouth Aspect Ratio (MAR) computation
  - Drowsiness state machine
  - Yawn detection
"""

import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from scipy.spatial import distance as dist


# -----------------------------------------------------------------------
# Driver state enum
# -----------------------------------------------------------------------

class DriverState(Enum):
    ALERT     = auto()
    DROWSY    = auto()
    YAWNING   = auto()
    EMERGENCY = auto()
    NO_FACE   = auto()


# -----------------------------------------------------------------------
# Metrics dataclass
# -----------------------------------------------------------------------

@dataclass
class DrowsinessMetrics:
    ear: float = 1.0
    mar: float = 0.0
    ear_consec_frames: int = 0
    yawn_consec_frames: int = 0
    eyes_closed_start: Optional[float] = None
    drowsy_start: Optional[float] = None
    state: DriverState = DriverState.ALERT
    emergency_countdown: float = 0.0
    alarm_triggered: bool = False
    emergency_triggered: bool = False
    total_blinks: int = 0
    total_yawns: int = 0
    session_start: float = field(default_factory=time.time)


# -----------------------------------------------------------------------
# Ratio helpers
# -----------------------------------------------------------------------

def compute_ear(eye_points: np.ndarray) -> float:
    """
    Eye Aspect Ratio (EAR).

    Uses 6 landmark points arranged as:
        p1 (left corner) → p2, p3, p4 (right corner) → p5, p6

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A value near 0.30 indicates open eyes; below ~0.25 → closed.
    """
    # Vertical distances
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Horizontal distance
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return float(ear)


def compute_mar(mouth_points: np.ndarray) -> float:
    """
    Mouth Aspect Ratio (MAR).

    Uses 8 simplified landmark points:
        [0]=left corner, [1]=right corner,
        [2]=top-outer, [3]=bottom-outer,
        [4]=top-left,  [5]=top-right,
        [6]=bottom-left, [7]=bottom-right

    MAR = vertical opening / horizontal width
    A value > 0.6 indicates a yawn / wide open mouth.
    """
    # Vertical distances (3 pairs)
    A = dist.euclidean(mouth_points[2], mouth_points[3])
    B = dist.euclidean(mouth_points[4], mouth_points[6])
    C = dist.euclidean(mouth_points[5], mouth_points[7])
    # Horizontal distance
    D = dist.euclidean(mouth_points[0], mouth_points[1])
    mar = (A + B + C) / (3.0 * D + 1e-6)
    return float(mar)


# -----------------------------------------------------------------------
# Drowsiness detector
# -----------------------------------------------------------------------

class DrowsinessDetector:
    """
    Stateful detector that maintains per-frame counters and timestamps
    to classify driver state.
    """

    def __init__(self, config: dict):
        t = config["thresholds"]
        self.EAR_THRESHOLD       = t["EAR_THRESHOLD"]
        self.MAR_THRESHOLD       = t["MAR_THRESHOLD"]
        self.EAR_CONSEC_FRAMES   = t["EAR_CONSEC_FRAMES"]
        self.YAWN_CONSEC_FRAMES  = t["YAWN_CONSEC_FRAMES"]
        self.DROWSY_TIME_SEC     = t["DROWSY_TIME_SECONDS"]
        self.EMERGENCY_TIME_SEC  = t["EMERGENCY_TIME_SECONDS"]

        self.metrics = DrowsinessMetrics(session_start=time.time())
        self._blink_in_progress  = False
        self._yawn_in_progress   = False

    # ------------------------------------------------------------------
    # Main update method (called once per frame)
    # ------------------------------------------------------------------

    def update(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        mouth: np.ndarray,
        face_detected: bool,
    ) -> DrowsinessMetrics:
        """
        Process one frame worth of landmarks and update internal state.

        Parameters
        ----------
        left_eye  : 6×2 array of left-eye landmark points
        right_eye : 6×2 array of right-eye landmark points
        mouth     : 8×2 array of mouth landmark points
        face_detected : whether a face was found in this frame

        Returns the updated DrowsinessMetrics object.
        """
        m = self.metrics
        now = time.time()

        if not face_detected:
            m.state = DriverState.NO_FACE
            # Reset eye-closed timer when face disappears
            m.eyes_closed_start = None
            m.ear_consec_frames = 0
            return m

        # ---- Compute ratios ----
        ear_left  = compute_ear(left_eye)
        ear_right = compute_ear(right_eye)
        m.ear = (ear_left + ear_right) / 2.0
        m.mar = compute_mar(mouth)

        # ---- Eye closure tracking ----
        if m.ear < self.EAR_THRESHOLD:
            m.ear_consec_frames += 1
            if m.eyes_closed_start is None:
                m.eyes_closed_start = now

            # Count blinks (short closures < 0.4s reset to alert)
            elapsed_closed = now - m.eyes_closed_start
        else:
            # Eyes opened — check if this was a blink
            if m.ear_consec_frames > 0 and m.eyes_closed_start is not None:
                closed_dur = now - m.eyes_closed_start
                if closed_dur < 0.4:
                    m.total_blinks += 1
                    self._blink_in_progress = False

            m.ear_consec_frames = 0
            m.eyes_closed_start = None
            elapsed_closed = 0.0

        # ---- Yawn tracking ----
        if m.mar > self.MAR_THRESHOLD:
            m.yawn_consec_frames += 1
            if m.yawn_consec_frames >= self.YAWN_CONSEC_FRAMES and not self._yawn_in_progress:
                self._yawn_in_progress = True
                m.total_yawns += 1
        else:
            m.yawn_consec_frames = 0
            self._yawn_in_progress = False

        # ---- State machine ----
        elapsed_closed = (now - m.eyes_closed_start) if m.eyes_closed_start else 0.0

        if m.ear_consec_frames >= self.EAR_CONSEC_FRAMES and elapsed_closed >= self.DROWSY_TIME_SEC:
            # Eyes closed long enough → DROWSY
            if m.drowsy_start is None:
                m.drowsy_start = now
            m.alarm_triggered = True

            drowsy_elapsed = now - m.drowsy_start
            remaining = max(0.0, self.EMERGENCY_TIME_SEC - drowsy_elapsed)
            m.emergency_countdown = remaining

            if drowsy_elapsed >= self.EMERGENCY_TIME_SEC:
                m.state = DriverState.EMERGENCY
                m.emergency_triggered = True
            else:
                m.state = DriverState.DROWSY

        elif m.yawn_consec_frames >= self.YAWN_CONSEC_FRAMES:
            m.state = DriverState.YAWNING
            m.drowsy_start = None
            m.alarm_triggered = False
            m.emergency_countdown = 0.0
        else:
            m.state = DriverState.ALERT
            m.drowsy_start = None
            m.alarm_triggered = False
            m.emergency_triggered = False
            m.emergency_countdown = 0.0

        return m

    def reset_emergency(self) -> None:
        """Call this after the emergency alert has been dispatched."""
        m = self.metrics
        m.emergency_triggered = False
        m.alarm_triggered = False
        m.drowsy_start = None
        m.eyes_closed_start = None
        m.ear_consec_frames = 0
        m.emergency_countdown = 0.0

    def get_session_summary(self) -> dict:
        m = self.metrics
        duration = time.time() - m.session_start
        return {
            "session_duration_sec": round(duration, 1),
            "total_blinks": m.total_blinks,
            "total_yawns": m.total_yawns,
            "emergency_triggered": m.emergency_triggered,
        }
