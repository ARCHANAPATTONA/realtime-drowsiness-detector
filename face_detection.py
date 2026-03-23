"""
face_detection.py
-----------------
Compatible with mediapipe >= 0.10.x (new Tasks API) AND older versions.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
MOUTH_IDX = [61, 291, 39, 181, 0, 17, 269, 405, 13, 14, 78, 308, 82, 87, 312, 317]
MOUTH_MAR_IDX = [61, 291, 0, 17, 78, 308, 82, 312]


class FaceDetector:
    def __init__(self, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._use_tasks = False
        try:
            import mediapipe as mp
            _ = mp.solutions.face_mesh  # test if solutions exists
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._use_tasks = False
            print("[FaceDetector] Using MediaPipe solutions API.")
        except AttributeError:
            import mediapipe as mp
            import os, urllib.request
            self._mp = mp
            self._use_tasks = True
            model_path = "face_landmarker.task"
            if not os.path.exists(model_path):
                url = ("https://storage.googleapis.com/mediapipe-models/"
                       "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
                print("[FaceDetector] Downloading face landmark model (~5 MB)...")
                urllib.request.urlretrieve(url, model_path)
                print("[FaceDetector] Model downloaded.")
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.face_mesh = vision.FaceLandmarker.create_from_options(options)
            print("[FaceDetector] Using MediaPipe Tasks API.")

    def process_frame(self, frame):
        annotated = frame.copy()
        h, w = frame.shape[:2]
        if self._use_tasks:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.face_mesh.detect(mp_image)
            if not results.face_landmarks:
                return None, annotated
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]]
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.face_mesh.process(rgb)
            rgb.flags.writeable = True
            if not results.multi_face_landmarks:
                return None, annotated
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
        return landmarks, annotated

    def get_eye_landmarks(self, landmarks):
        left  = np.array([landmarks[i] for i in LEFT_EYE_IDX],  dtype=np.float32)
        right = np.array([landmarks[i] for i in RIGHT_EYE_IDX], dtype=np.float32)
        return left, right

    def get_mouth_landmarks(self, landmarks):
        return np.array([landmarks[i] for i in MOUTH_IDX], dtype=np.float32)

    def get_mouth_mar_landmarks(self, landmarks):
        return np.array([landmarks[i] for i in MOUTH_MAR_IDX], dtype=np.float32)

    def draw_eye_landmarks(self, frame, left_eye, right_eye, color=(0,255,0), thickness=1):
        cv2.polylines(frame, [left_eye.astype(np.int32)],  True, color, thickness)
        cv2.polylines(frame, [right_eye.astype(np.int32)], True, color, thickness)

    def draw_mouth_landmarks(self, frame, mouth, color=(0,165,255), thickness=1):
        cv2.polylines(frame, [mouth.astype(np.int32)], True, color, thickness)

    def draw_face_box(self, frame, landmarks, color=(255,255,255), thickness=1):
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), color, thickness)

    def release(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass