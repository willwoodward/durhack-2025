import mediapipe as mp
from typing import Optional
import numpy as np

from . import EventDetector, DetectionEvent
from src.consts import SWIPE_COOLDOWN, SWIPE_MIN_DISTANCE, SWIPE_STABILITY_FRAMES, SWIPE_VELOCITY_THRESHOLD

class SwipeDetector(EventDetector):
    """
    Detects horizontal swipe gestures with a hand.
    Tracks sustained movement in one direction (left-to-right or right-to-left).
    """

    def __init__(self, hand: str, direction: str):
        """
        :param hand: "left" or "right"
        :param direction: "left_to_right" or "right_to_left"
        """
        self.hand = hand
        self.direction = direction

        name = f"{hand}_swipe_{direction}"
        super().__init__(name, cooldown=SWIPE_COOLDOWN)

        # Configuration
        self.velocity_threshold = SWIPE_VELOCITY_THRESHOLD
        self.min_distance = SWIPE_MIN_DISTANCE
        self.stability_frames = SWIPE_STABILITY_FRAMES
        self.max_other_hand_visibility = 0.6  # Block swipes if other hand is visible

        # State tracking
        self.prev_position = None
        self.prev_time = None
        self.swipe_frame_count = 0
        self.start_position = None
        self.total_distance = 0
        self.onset_time = None

        # Debug info
        self.current_velocity = 0
        self.just_detected = False

        # Wrist landmarks
        if hand == "left":
            self.wrist_id = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
            self.other_wrist_id = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
        else:
            self.wrist_id = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value
            self.other_wrist_id = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        if not self.can_detect(frame_time) or landmarks is None:
            self.just_detected = False
            return None

        wrist = landmarks[self.wrist_id]
        other_wrist = landmarks[self.other_wrist_id]

        # Check visibility of swiping hand
        if not self.check_visibility(wrist):
            self._reset_swipe()
            return None

        # Block swipes if both hands are visible (prevents swipes during claps)
        if other_wrist.visibility >= self.max_other_hand_visibility:
            self._reset_swipe()
            return None

        current_position = np.array([wrist.x, wrist.y, wrist.z])

        if self.prev_position is not None and self.prev_time is not None:
            dt = frame_time - self.prev_time
            if dt > 0 and dt < 0.5:  # Ignore large time gaps
                # Calculate horizontal velocity (x-axis)
                displacement = current_position - self.prev_position
                horizontal_velocity = displacement[0] / dt  # Positive = right, negative = left
                self.current_velocity = horizontal_velocity

                # Check if moving in the correct direction with enough speed
                moving_correctly = False
                if self.direction == "left_to_right" and horizontal_velocity > self.velocity_threshold:
                    moving_correctly = True
                elif self.direction == "right_to_left" and horizontal_velocity < -self.velocity_threshold:
                    moving_correctly = True

                if moving_correctly:
                    # Start tracking if this is the first frame
                    if self.swipe_frame_count == 0:
                        self.start_position = self.prev_position.copy()
                        self.total_distance = 0
                        self.onset_time = frame_time

                    self.swipe_frame_count += 1
                    self.total_distance += abs(displacement[0])
                else:
                    # Reset if movement stopped or changed direction
                    self._reset_swipe()

                # Trigger detection if stable enough and traveled far enough
                if (self.swipe_frame_count >= self.stability_frames and
                    self.total_distance >= self.min_distance):
                    onset = self.onset_time if self.onset_time is not None else frame_time
                    offset = frame_time
                    self.record_detection(frame_time)
                    self.just_detected = True
                    event = DetectionEvent(
                        name=self.name,
                        timestamp=frame_time,
                        confidence=min(1.0, self.total_distance / (self.min_distance * 2)),
                        onset_time=onset,
                        offset_time=offset,
                        metadata={
                            "hand": self.hand,
                            "direction": self.direction,
                            "distance": self.total_distance,
                            "velocity": horizontal_velocity
                        }
                    )
                    self._reset_swipe()
                    self.prev_position = current_position
                    self.prev_time = frame_time
                    return event

        self.prev_position = current_position
        self.prev_time = frame_time
        return None

    def _reset_swipe(self):
        """Reset swipe tracking state."""
        self.swipe_frame_count = 0
        self.start_position = None
        self.total_distance = 0
        self.onset_time = None
        self.just_detected = False