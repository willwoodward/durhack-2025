import mediapipe as mp
from typing import Optional
from . import EventDetector, DetectionEvent
from src.consts import CHOP_COOLDOWN, CHOP_VELOCITY_THRESHOLD, CHOP_STABILITY_FRAMES

class ChopDetector(EventDetector):
    """Detects karate chop gestures with each hand."""

    def __init__(self):
        super().__init__("chop", cooldown=CHOP_COOLDOWN)

        # Detection parameters
        self.velocity_threshold = CHOP_VELOCITY_THRESHOLD
        self.stability_frames = CHOP_STABILITY_FRAMES
        self.orientation_threshold = 0.5  # fraction to detect roughly vertical palm

        # State tracking per hand
        self.prev_positions = {"left": None, "right": None}
        self.close_frame_count = {"left": 0, "right": 0}

        # Debug info
        self.current_speed = {"left": 0, "right": 0}
        self.current_orientation = {"left": 0, "right": 0}
        self.just_detected = False

    def _get_hand_landmarks(self, landmarks):
        """Extract left and right wrist landmarks."""
        left_wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST.value]
        right_wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST.value]
        return left_wrist, right_wrist

    def _calculate_velocity(self, hand_label, hand_landmark):
        """Compute horizontal hand velocity between frames."""
        x = hand_landmark.x
        prev = self.prev_positions[hand_label]
        if prev:
            vx = x - prev
        else:
            vx = 0.0
        self.prev_positions[hand_label] = x
        self.current_speed[hand_label] = abs(vx)
        return abs(vx)

    def _check_orientation(self, hand_landmarks):
        """Estimate if hand is roughly vertical using wrist->index vector."""
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        dx = index_mcp.x - wrist.x
        dy = index_mcp.y - wrist.y
        # fraction of horizontal vs total vector length
        orientation = abs(dx) / math.sqrt(dx*dx + dy*dy)
        return orientation  # near 0 = vertical, near 1 = horizontal

    def _update_state(self, hand_label, velocity, orientation):
        """Track consecutive frames of chop-like motion."""
        self.current_orientation[hand_label] = orientation
        if velocity > self.velocity_threshold and orientation < self.orientation_threshold:
            self.close_frame_count[hand_label] += 1
        else:
            self.close_frame_count[hand_label] = 0

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        if not self.can_detect(frame_time):
            self.close_frame_count["left"] = 0
            self.close_frame_count["right"] = 0
            self.just_detected = False
            return None

        chop_detected = False

        for hand_label in ["left", "right"]:
            # Extract wrist landmark
            hand_landmark = landmarks[mp.solutions.hands.HandLandmark.WRIST.value]

            # Compute velocity and orientation
            velocity = self._calculate_velocity(hand_label, hand_landmark)
            orientation = self._check_orientation(landmarks)

            # Update state
            self._update_state(hand_label, velocity, orientation)

            if self.close_frame_count[hand_label] >= self.stability_frames:
                chop_detected = True
                self.close_frame_count[hand_label] = 0  # reset after detection

        if chop_detected:
            self.just_detected = True
            self.record_detection(frame_time)
            return DetectionEvent(
                name=self.name,
                timestamp=frame_time,
                confidence=1.0,  # could scale by velocity or orientation
                onset_time=frame_time,
                offset_time=frame_time,
                metadata={
                    "speed": self.current_speed,
                    "orientation": self.current_orientation
                }
            )
        else:
            self.just_detected = False

        return None




