from typing import Optional
import mediapipe as mp
import numpy as np

from . import EventDetector, DetectionEvent
from src.consts import CHOP_COOLDOWN

class ChopDetector(EventDetector):
    """
    Detects elbow flick motion.
    """

    def __init__(self, hand: str):
      self.hand = hand
      name = "left_chop" if hand == "left" else "right_chop"
      super().__init__(name, cooldown=CHOP_COOLDOWN)

      
