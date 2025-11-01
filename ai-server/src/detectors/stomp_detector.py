from typing import Optional
import numpy as np
import mediapipe as mp

from . import DetectionEvent, EventDetector
from src.consts import STOMP_COOLDOWN, STOMP_STABILITY_FRAMES, STOMP_VELOCITY_THRESHOLD, MIN_VISIBILITY

class StompDetector(EventDetector):
    """Detects foot stomping (rapid downward foot movement)."""

    def __init__(self):
        super().__init__("stomp", cooldown=STOMP_COOLDOWN)
        self.velocity_threshold = STOMP_VELOCITY_THRESHOLD
        self.prev_foot_positions = None
        self.prev_time = None
        self.stability_frames = STOMP_STABILITY_FRAMES
        self.high_velocity_count = 0
        self.onset_time = None  # When high velocity first detected

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        # Get foot landmarks
        left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

        # Check visibility - skip if feet aren't visible
        if left_ankle.visibility < MIN_VISIBILITY and right_ankle.visibility < MIN_VISIBILITY:
            return None

        current_positions = {
            'left': np.array([left_ankle.x, left_ankle.y, left_ankle.z]),
            'right': np.array([right_ankle.x, right_ankle.y, right_ankle.z])
        }

        detected_foot = None
        max_velocity = 0

        if self.prev_foot_positions is not None and self.prev_time is not None:
            dt = frame_time - self.prev_time
            if dt > 0 and dt < 0.5:  # Ignore large time gaps
                # Check each foot for stomp (only if visible)
                for foot_name, current_pos in current_positions.items():
                    # Check if this specific foot is visible
                    ankle = left_ankle if foot_name == 'left' else right_ankle
                    if ankle.visibility < MIN_VISIBILITY:
                        continue

                    prev_pos = self.prev_foot_positions[foot_name]

                    # Calculate vertical velocity (y-axis, positive is downward)
                    vertical_velocity = (current_pos[1] - prev_pos[1]) / dt

                    if vertical_velocity > self.velocity_threshold:
                        if vertical_velocity > max_velocity:
                            max_velocity = vertical_velocity
                            detected_foot = foot_name

        # Track stability
        if detected_foot and max_velocity > self.velocity_threshold:
            # Track onset time when high velocity first detected
            if self.high_velocity_count == 0:
                self.onset_time = frame_time
            self.high_velocity_count += 1
        else:
            self.high_velocity_count = 0
            self.onset_time = None

        # Only trigger after seeing high velocity for multiple frames
        if self.high_velocity_count >= self.stability_frames and self.can_detect(frame_time):
            onset = self.onset_time if self.onset_time is not None else frame_time
            offset = frame_time
            self.record_detection(frame_time)
            self.high_velocity_count = 0
            self.onset_time = None
            self.prev_foot_positions = current_positions
            self.prev_time = frame_time
            return DetectionEvent(
                name=self.name,
                timestamp=frame_time,
                confidence=min(1.0, max_velocity / (self.velocity_threshold * 2)),
                onset_time=onset,
                offset_time=offset,
                metadata={"foot": detected_foot, "velocity": max_velocity}
            )

        self.prev_foot_positions = current_positions
        self.prev_time = frame_time
        return None