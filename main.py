"""
main.py
-------
Driver Drowsiness Detection and Emergency Alert System
Entry point: initialises camera, runs the detection loop, renders the HUD.

Usage
-----
    python main.py                  # uses config.json in same directory
    python main.py --config path/to/config.json
    python main.py --camera 1       # select a different camera index
    python main.py --no-alert       # disable sound (useful for testing)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from face_detection    import FaceDetector
from drowsiness_detection import DrowsinessDetector, DriverState
from alert_system      import AlertSystem

# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# -----------------------------------------------------------------------
# HUD / Overlay rendering
# -----------------------------------------------------------------------

# Palette (BGR)
COLOR_GREEN  = (0,   220,   0)
COLOR_ORANGE = (0,   165, 255)
COLOR_YELLOW = (0,   220, 220)
COLOR_RED    = (0,     0, 255)
COLOR_GREY   = (140, 140, 140)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,     0,   0)
COLOR_DARK   = (20,   20,  20)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _put_text(
    frame, text, pos, color=COLOR_WHITE,
    scale=0.65, thickness=2, shadow=True
):
    """Draw text with optional drop-shadow for readability."""
    if shadow:
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                    FONT, scale, COLOR_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    metrics,
    fps: float,
    config: dict,
) -> np.ndarray:
    """
    Render all HUD elements on top of the live feed.
    Returns the annotated frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ---- Top status bar ----
    STATE_COLOR = {
        DriverState.ALERT    : COLOR_GREEN,
        DriverState.DROWSY   : COLOR_ORANGE,
        DriverState.YAWNING  : COLOR_YELLOW,
        DriverState.EMERGENCY: COLOR_RED,
        DriverState.NO_FACE  : COLOR_GREY,
    }
    STATE_LABEL = {
        DriverState.ALERT    : "  Driver Alert  ",
        DriverState.DROWSY   : "  Drowsy!  ",
        DriverState.YAWNING  : "  Yawning  ",
        DriverState.EMERGENCY: "  EMERGENCY ALERT SENT  ",
        DriverState.NO_FACE  : "  No Face Detected  ",
    }

    bar_color = STATE_COLOR.get(metrics.state, COLOR_GREY)
    bar_label = STATE_LABEL.get(metrics.state, "")

    # Transparent status bar
    cv2.rectangle(overlay, (0, 0), (w, 52), bar_color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # Re-draw label on the blended result
    (tw, _), _ = cv2.getTextSize(bar_label, FONT_BOLD, 0.95, 2)
    cv2.putText(frame, bar_label, ((w - tw) // 2, 36),
                FONT_BOLD, 0.95, COLOR_WHITE, 2, cv2.LINE_AA)

    # ---- Bottom metrics panel ----
    panel_h = 110
    panel = frame[h - panel_h : h, 0 : w].copy()
    cv2.rectangle(panel, (0, 0), (w, panel_h), COLOR_DARK, -1)
    cv2.addWeighted(panel, 0.70, frame[h - panel_h : h, 0 : w], 0.30, 0, frame[h - panel_h : h, 0 : w])

    y0 = h - panel_h + 22
    _put_text(frame, f"EAR : {metrics.ear:.3f}", (14, y0),             COLOR_WHITE, 0.60, 1)
    _put_text(frame, f"MAR : {metrics.mar:.3f}", (14, y0 + 28),        COLOR_WHITE, 0.60, 1)
    _put_text(frame, f"FPS : {fps:.1f}",          (14, y0 + 56),        COLOR_WHITE, 0.60, 1)

    ear_thr = config["thresholds"]["EAR_THRESHOLD"]
    mar_thr = config["thresholds"]["MAR_THRESHOLD"]
    _put_text(frame, f"EAR threshold: {ear_thr}",  (200, y0),           COLOR_GREY, 0.55, 1)
    _put_text(frame, f"MAR threshold: {mar_thr}",  (200, y0 + 28),      COLOR_GREY, 0.55, 1)
    _put_text(frame, f"Blinks: {metrics.total_blinks}  Yawns: {metrics.total_yawns}",
              (200, y0 + 56), COLOR_GREY, 0.55, 1)

    # ---- Emergency countdown ----
    if metrics.state == DriverState.DROWSY and metrics.emergency_countdown > 0:
        cnt = metrics.emergency_countdown
        pct = cnt / config["thresholds"]["EMERGENCY_TIME_SECONDS"]

        cx, cy, r = w - 70, h - panel_h // 2, 38
        # Background circle
        cv2.circle(frame, (cx, cy), r, COLOR_DARK, -1)
        cv2.circle(frame, (cx, cy), r, COLOR_ORANGE, 2)

        # Arc showing remaining time
        start_angle = -90
        end_angle   = start_angle - int(360 * (1 - pct))
        cv2.ellipse(frame, (cx, cy), (r, r), 0,
                    start_angle, end_angle, COLOR_RED, 3, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(f"{cnt:.0f}s", FONT_BOLD, 0.65, 2)
        cv2.putText(frame, f"{cnt:.0f}s",
                    (cx - tw // 2, cy + th // 2),
                    FONT_BOLD, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

        _put_text(frame, "Emergency in:", (w - 165, h - panel_h + 18), COLOR_ORANGE, 0.50, 1)

    # ---- Emergency flash overlay ----
    if metrics.state == DriverState.EMERGENCY:
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), COLOR_RED, -1)
        alpha = 0.18 + 0.10 * abs(np.sin(time.time() * 5))
        cv2.addWeighted(flash, alpha, frame, 1 - alpha, 0, frame)
        _put_text(frame, "EMERGENCY ALERT SENT", (w // 2 - 220, h // 2),
                  COLOR_WHITE, 1.2, 3)
        _put_text(frame, "Help has been notified.", (w // 2 - 165, h // 2 + 50),
                  COLOR_WHITE, 0.8, 2)

    return frame


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

def run(config: dict, no_alert: bool = False, camera_idx: Optional[int] = None) -> None:
    cam_idx = camera_idx if camera_idx is not None else config["camera"]["device_index"]

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {cam_idx}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["camera"]["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["frame_height"])
    cap.set(cv2.CAP_PROP_FPS,          config["camera"]["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimise latency

    detector  = FaceDetector()
    drowsy    = DrowsinessDetector(config)
    alerts    = AlertSystem(config)

    if no_alert:
        logger.info("Alert sound disabled (--no-alert flag).")

    logger.info("System running. Press 'q' to quit, 'r' to reset emergency state.")

    # FPS counter
    fps      = 0.0
    fps_buf  = []
    t_prev   = time.time()

    window_name = "Driver Drowsiness Detection System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame grab failed — retrying.")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)   # mirror for natural UX

        # ---- FPS ----
        now = time.time()
        fps_buf.append(1.0 / max(now - t_prev, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps   = sum(fps_buf) / len(fps_buf)
        t_prev = now

        # ---- Detection ----
        landmarks, frame = detector.process_frame(frame)
        face_found = landmarks is not None

        if face_found:
            left_eye, right_eye = detector.get_eye_landmarks(landmarks)
            mouth_pts           = detector.get_mouth_mar_landmarks(landmarks)
            mouth_full          = detector.get_mouth_landmarks(landmarks)

            # Draw landmarks
            eye_color   = tuple(config["ui"]["landmark_color_eyes"])
            mouth_color = tuple(config["ui"]["landmark_color_mouth"])
            detector.draw_eye_landmarks(frame, left_eye, right_eye, eye_color, 1)
            detector.draw_mouth_landmarks(frame, mouth_full, mouth_color, 1)
            detector.draw_face_box(frame, landmarks, (200, 200, 200), 1)

            metrics = drowsy.update(left_eye, right_eye, mouth_pts, True)
        else:
            metrics = drowsy.update(None, None, None, False)

        # ---- Alerts ----
        if not no_alert:
            if metrics.alarm_triggered:
                alerts.trigger_alarm()
            else:
                alerts.stop_alarm()

        if metrics.emergency_triggered:
            alerts.send_emergency("UNRESPONSIVE/DROWSY")

        # ---- HUD ----
        frame = draw_hud(frame, metrics, fps, config)

        cv2.imshow(window_name, frame)

        # ---- Keyboard ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            logger.info("Manual reset triggered.")
            drowsy.reset_emergency()
            alerts.reset()

    # ---- Cleanup ----
    cap.release()
    detector.release()
    alerts.stop_alarm()
    cv2.destroyAllWindows()

    summary = drowsy.get_session_summary()
    logger.info(f"Session summary: {summary}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    p.add_argument("--config",   default="config.json", help="Path to config.json")
    p.add_argument("--camera",   type=int, default=None, help="Camera device index")
    p.add_argument("--no-alert", action="store_true",   help="Disable alarm sound")
    return p.parse_args()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        return json.load(f)


if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)

    try:
        run(config, no_alert=args.no_alert, camera_idx=args.camera)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
