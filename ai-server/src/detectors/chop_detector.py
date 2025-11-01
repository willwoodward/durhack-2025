from typing import Optional
import mediapipe as mp
import numpy as np

from . import EventDetector, DetectionEvent
from src.consts import CHOP_COOLDOWN, CHOP_DISTANCE_THRESHOLD, CHOP_DISTANCE_APART, CHOP_STABILITY_FRAMES

class ChopDetector(EventDetector):
    """
    Detects elbow flick motion.
    """

    def __init__(self, hand: str):
      self.hand = hand
      name = "left_chop" if hand == "left" else "right_chop"
      super().__init__(name, cooldown=CHOP_COOLDOWN)
      self.distance_threshold = CHOP_DISTANCE_THRESHOLD
      self.distance_apart = CHOP_DISTANCE_APART
      self.was_apart = True # Track if hand is above elbow
      self.stability_frames = CHOP_STABILITY_FRAMES
      self.close_frame_count = 0
      self.apart_frame_count = 0

      # debug info
      self.current_distance = 0
      self.current_state = "unknown"
      self.just_detected = False

      # Onset/offset tracking
      self.onset_time = None # When hand gets close to level with elbow

      # fingertips indices
      self.finger_tip_ids = [
          mp.solutions.hands.HandLandmark.THUMB_TIP,
          mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
          mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
          mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
          mp.solutions.hands.HandLandmark.PINKY_TIP,
      ]

      # elbow landmark index (for rotation vector)
      if hand == "left":
          self.elbow_id = mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value
          self.wrist_id = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
      else:
          self.elbow_id = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value
          self.wrist_id = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value

    def get_wrist_height(self, landmarks):
        wrist = landmarks[self.wrist_id]
        return wrist.y

    def get_elbow_height(self, landmarks):
        elbow = landmarks[self.elbow_id]
        return elbow.y
    
    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        if not self.can_detect(frame_time) or landmarks is None:
            return None
        



