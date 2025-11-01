from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from src.consts import MIN_VISIBILITY

@dataclass
class DetectionEvent:
    """Represents a detected event."""
    name: str
    timestamp: float
    confidence: float
    onset_time: float  # When the gesture started
    offset_time: float  # When the gesture was confirmed/completed
    metadata: dict = None


class EventDetector(ABC):
    """Base class for all event detectors."""

    def __init__(self, name: str, cooldown: float = 0.5):
        """
        Initialize event detector.

        Args:
            name: Name of the event
            cooldown: Minimum time (seconds) between detections to avoid duplicates
        """
        self.name = name
        self.cooldown = cooldown
        self.last_detection_time = 0

    @abstractmethod
    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        """
        Detect event from pose landmarks.

        Args:
            landmarks: MediaPipe pose landmarks
            frame_time: Current frame timestamp

        Returns:
            DetectionEvent if detected, None otherwise
        """
        pass

    def can_detect(self, frame_time: float) -> bool:
        """Check if enough time has passed since last detection."""
        return (frame_time - self.last_detection_time) >= self.cooldown

    def record_detection(self, frame_time: float):
        """Record that a detection occurred."""
        self.last_detection_time = frame_time

    # Helper methods for common detection patterns
    def check_threshold_stable(self, value: float, threshold: float,
                               below: bool, frame_counter: int,
                               required_frames: int) -> Tuple[int, bool]:
        """
        Check if a value stays below/above a threshold for consecutive frames.

        Args:
            value: Current value to check
            threshold: Threshold to compare against
            below: If True, check if value < threshold; if False, check if value > threshold
            frame_counter: Current count of consecutive frames meeting condition
            required_frames: Number of consecutive frames required

        Returns:
            Tuple of (new_frame_counter, condition_met)
        """
        if below:
            condition_met = value < threshold
        else:
            condition_met = value > threshold

        if condition_met:
            frame_counter += 1
        else:
            frame_counter = 0

        return frame_counter, (frame_counter >= required_frames)

    def calculate_distance_3d(self, point1, point2) -> float:
        """
        Calculate 3D Euclidean distance between two landmarks.

        Args:
            point1: First landmark with x, y, z coordinates
            point2: Second landmark with x, y, z coordinates

        Returns:
            Distance as float
        """
        return np.sqrt(
            (point1.x - point2.x)**2 +
            (point1.y - point2.y)**2 +
            (point1.z - point2.z)**2
        )

    def check_visibility(self, *landmarks, min_visibility: float = MIN_VISIBILITY) -> bool:
        """
        Check if all provided landmarks meet minimum visibility threshold.

        Args:
            *landmarks: Variable number of landmarks to check
            min_visibility: Minimum visibility threshold (0-1)

        Returns:
            True if all landmarks are visible enough, False otherwise
        """
        return all(landmark.visibility >= min_visibility for landmark in landmarks)