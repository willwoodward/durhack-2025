import mediapipe as mp
from typing import Optional

from . import EventDetector, DetectionEvent

from src.consts import CLAP_COOLDOWN, CLAP_DISTANCE_APART, CLAP_DISTANCE_THRESHOLD, CLAP_STABILITY_FRAMES

class ClapDetector(EventDetector):
    """Detects hand clapping gestures."""

    def __init__(self):
        super().__init__("clap", cooldown=CLAP_COOLDOWN)
        self.distance_threshold = CLAP_DISTANCE_THRESHOLD
        self.distance_apart = CLAP_DISTANCE_APART
        self.was_apart = True  # Track if hands were apart
        self.stability_frames = CLAP_STABILITY_FRAMES
        self.close_frame_count = 0  # Count consecutive frames hands are close
        self.apart_frame_count = 0  # Count consecutive frames hands are apart

        # Debug info
        self.current_distance = 0
        self.current_state = "unknown"  # "close", "apart", "medium", or "low_visibility"
        self.just_detected = False  # Flag to show green flash on detection

        # Onset/offset tracking
        self.onset_time = None  # When hands first got close

    def _get_hand_landmarks(self, landmarks):
        """Extract left and right wrist landmarks."""
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        return left_wrist, right_wrist

    def _are_hands_apart(self, distance: float) -> bool:
        """Check if hands are far enough apart to be considered separated."""
        return distance > self.distance_apart

    def _are_hands_close(self, distance: float) -> bool:
        """Check if hands are close enough to potentially be clapping."""
        return distance < self.distance_threshold

    def _update_hand_state(self, distance: float, frame_time: float) -> None:
        """
        Update frame counters and state based on current hand distance.

        This tracks whether hands are consistently close or apart across frames,
        which is essential for detecting a deliberate clap vs. random movement.
        """
        if self._are_hands_close(distance):
            # Track onset time when hands first get close
            if self.close_frame_count == 0:
                self.onset_time = frame_time
            self.close_frame_count += 1
            self.apart_frame_count = 0
            self.current_state = "close"
        elif self._are_hands_apart(distance):
            self.apart_frame_count += 1
            self.close_frame_count = 0
            self.current_state = "apart"
            self.onset_time = None  # Reset onset time when hands move apart
        else:
            # In-between state, don't reset counters completely
            self.current_state = "medium"

        # Update was_apart flag if hands have been consistently apart
        if self.apart_frame_count >= self.stability_frames:
            self.was_apart = True

    def _reset_state(self, state: str = "unknown") -> None:
        """Reset frame counters and set current state."""
        self.close_frame_count = 0
        self.apart_frame_count = 0
        self.current_state = state
        self.just_detected = False
        self.onset_time = None

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        if not self.can_detect(frame_time):
            self._reset_state("cooldown")
            return None

        # Get hand landmarks
        left_wrist, right_wrist = self._get_hand_landmarks(landmarks)

        # Check visibility - reset counters if hands aren't visible
        if not self.check_visibility(left_wrist, right_wrist):
            # Reset counters when visibility drops to prevent false detections from jumping landmarks
            self._reset_state("low_visibility")
            return None

        # Calculate distance between hands using helper
        distance = self.calculate_distance_3d(left_wrist, right_wrist)
        self.current_distance = distance

        # Update frame counters and state
        self._update_hand_state(distance, frame_time)

        # Detect clap: hands close for multiple frames AND were previously apart
        if self.close_frame_count >= self.stability_frames and self.was_apart:
            onset = self.onset_time if self.onset_time is not None else frame_time
            offset = frame_time
            self.was_apart = False
            self.close_frame_count = 0
            self.apart_frame_count = 0
            self.just_detected = True
            self.record_detection(frame_time)
            return DetectionEvent(
                name=self.name,
                timestamp=frame_time,
                confidence=1.0 - (distance / self.distance_threshold),
                onset_time=onset,
                offset_time=offset,
                metadata={"distance": distance}
            )
        else:
            self.just_detected = False

        return None