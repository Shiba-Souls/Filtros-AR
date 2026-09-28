import ctypes
import sys
import cv2
import mediapipe as mp
import numpy as np

# Workaround for MediaPipe C bindings on Python 3.13+ on Windows
if sys.platform == "win32" and sys.version_info >= (3, 13):
    try:
        _free_func = ctypes.cdll.msvcrt.free
        _orig_getattr = ctypes.CDLL.__getattr__
        ctypes.CDLL.__getattr__ = lambda self, name: _free_func if name == "free" else _orig_getattr(self, name)
    except Exception:
        pass

# MediaPipe Tasks aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Landmark indices matching the legacy Mediapipe hand landmarks
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", max_num_hands=2, min_detection_confidence=0.3):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr):
        """Converts BGR image to MediaPipe Image format and runs hand detection."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect(mp_image)

    def draw_landmarks(self, frame_bgr, result):
        """Draws detected hand landmarks and connections manually onto the frame."""
        if not result.hand_landmarks:
            return frame_bgr

        h, w, _ = frame_bgr.shape
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

        for hand_landmarks in result.hand_landmarks:
            # Draw connections
            for conn in connections:
                pt1 = hand_landmarks[conn.start]
                pt2 = hand_landmarks[conn.end]
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw landmark points
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)

        return frame_bgr

    def close(self):
        self.landmarker.close()