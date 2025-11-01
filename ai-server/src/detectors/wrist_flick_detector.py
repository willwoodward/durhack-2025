from typing import Optional
import mediapipe as mp
import numpy as np

from . import EventDetector, DetectionEvent
from src.consts import FLICK_COOLDOWN, FLICK_STABILITY_FRAMES, FLICK_VELOCITY_THRESHOLD

class WristFlickDetector(EventDetector):
    """
    Detects quick wrist flicking motion using wrist + finger + forearm rotation.
    Combines linear velocity of hand center and angular velocity of wrist vector.
    """

    def __init__(self, hand: str):
        self.hand = hand
        name = "left_hand_flick" if hand == "left" else "right_hand_flick"
        super().__init__(name, cooldown=FLICK_COOLDOWN)

        # thresholds
        self.velocity_threshold = FLICK_VELOCITY_THRESHOLD
        self.angular_threshold = 0.6  # Min angular change (radians/frame) to count as flick
        self.stability_frames = FLICK_STABILITY_FRAMES

        # state
        self.prev_position = None
        self.prev_wrist_vector = None  # vector along forearm (wrist -> elbow)
        self.prev_time = None
        self.flick_frame_count = 0

        # debug
        self.current_speed = 0
        self.current_angular_velocity = 0
        self.just_detected = False

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

    def get_hand_center(self, landmarks):
        """Average wrist + fingertips to get smooth hand center."""
        wrist = landmarks[self.wrist_id]
        points = [wrist]
        for tip_id in self.finger_tip_ids:
            points.append(landmarks[tip_id])
        xyz = np.array([[p.x, p.y, p.z] for p in points])
        return np.mean(xyz, axis=0)

    def get_wrist_vector(self, landmarks):
        """Vector along forearm: wrist -> elbow."""
        wrist = landmarks[self.wrist_id]
        elbow = landmarks[self.elbow_id]
        return np.array([elbow.x - wrist.x, elbow.y - wrist.y, elbow.z - wrist.z])

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        if not self.can_detect(frame_time) or landmarks is None:
            return None

        # Hand center for linear velocity
        current_position = self.get_hand_center(landmarks)
        wrist_vector = self.get_wrist_vector(landmarks)

        linear_speed = 0
        angular_velocity = 0

        if self.prev_position is not None and self.prev_wrist_vector is not None and self.prev_time is not None:
            dt = frame_time - self.prev_time
            if dt > 0 and dt < 0.5:
                # linear velocity
                displacement = current_position - self.prev_position
                linear_speed = np.linalg.norm(displacement) / dt
                self.current_speed = linear_speed

                # angular velocity (angle change between forearm vectors)
                prev_vec_norm = self.prev_wrist_vector / (np.linalg.norm(self.prev_wrist_vector) + 1e-6)
                curr_vec_norm = wrist_vector / (np.linalg.norm(wrist_vector) + 1e-6)
                dot = np.clip(np.dot(prev_vec_norm, curr_vec_norm), -1.0, 1.0)
                angle_change = np.arccos(dot)  # radians/frame
                angular_velocity = angle_change / dt
                self.current_angular_velocity = angular_velocity

                # check if flick: high speed + high angular velocity
                if linear_speed > self.velocity_threshold and angular_velocity > self.angular_threshold:
                    self.flick_frame_count += 1
                else:
                    self.flick_frame_count = 0
                    self.just_detected = False

                # trigger detection if stable enough
                if self.flick_frame_count >= self.stability_frames:
                    self.record_detection(frame_time)
                    self.flick_frame_count = 0
                    self.just_detected = True
                    self.prev_position = current_position
                    self.prev_wrist_vector = wrist_vector
                    self.prev_time = frame_time
                    return DetectionEvent(
                        name=self.name,
                        timestamp=frame_time,
                        confidence=min(1.0, linear_speed / (self.velocity_threshold * 2)),
                        onset_time=frame_time,
                        offset_time=frame_time,
                        metadata={
                            "hand": self.hand,
                            "speed": linear_speed,
                            "angular_velocity": angular_velocity
                        }
                    )

        # update previous
        self.prev_position = current_position
        self.prev_wrist_vector = wrist_vector
        self.prev_time = frame_time
        return None