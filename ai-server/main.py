#!/usr/bin/env python3
"""
Pose estimation and event detection system using MediaPipe.
Detects events like hand clapping, foot stomping, and wrist flicks.
"""

import cv2
import mediapipe as mp
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
import time
import sys


# ==================== CONFIGURATION ====================
# Adjust these values to tune detection sensitivity

# Clap Detection
CLAP_DISTANCE_THRESHOLD = 0.5      # Max distance between hands to detect clap (lower = stricter)
CLAP_DISTANCE_APART = 0.5          # Min distance to consider hands "apart" before clapping
CLAP_STABILITY_FRAMES = 3           # Consecutive frames needed to confirm clap
CLAP_COOLDOWN = 0.5                 # Seconds between clap detections

# Stomp Detection
STOMP_VELOCITY_THRESHOLD = 0.5      # Min downward velocity to detect stomp (higher = stricter)
STOMP_STABILITY_FRAMES = 2          # Consecutive frames needed to confirm stomp
STOMP_COOLDOWN = 0.5                # Seconds between stomp detections

# Wrist Flick Detection (quick snapping motion)
FLICK_VELOCITY_THRESHOLD = 0.8      # Min linear speed of hand center to detect a flick
FLICK_DIRECTION_CHANGE = 0.5        # Max dot product between consecutive velocity vectors
FLICK_MIN_HAND_DISTANCE = 0.4       # Min distance between hands to avoid triggering on claps
FLICK_STABILITY_FRAMES = 1          # Number of consecutive frames motion must meet thresholds
FLICK_COOLDOWN = 0.15               # Minimum seconds between flick detections


# Visibility Thresholds
MIN_VISIBILITY = 0.5                # Min landmark visibility to detect events (0-1)

# =======================================================


@dataclass
class DetectionEvent:
    """Represents a detected event."""
    name: str
    timestamp: float
    confidence: float
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

    def _update_hand_state(self, distance: float) -> None:
        """
        Update frame counters and state based on current hand distance.

        This tracks whether hands are consistently close or apart across frames,
        which is essential for detecting a deliberate clap vs. random movement.
        """
        if self._are_hands_close(distance):
            self.close_frame_count += 1
            self.apart_frame_count = 0
            self.current_state = "close"
        elif self._are_hands_apart(distance):
            self.apart_frame_count += 1
            self.close_frame_count = 0
            self.current_state = "apart"
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
        self._update_hand_state(distance)

        # Detect clap: hands close for multiple frames AND were previously apart
        if self.close_frame_count >= self.stability_frames and self.was_apart:
            self.was_apart = False
            self.close_frame_count = 0
            self.apart_frame_count = 0
            self.just_detected = True
            self.record_detection(frame_time)
            return DetectionEvent(
                name=self.name,
                timestamp=frame_time,
                confidence=1.0 - (distance / self.distance_threshold),
                metadata={"distance": distance}
            )
        else:
            self.just_detected = False

        return None


class StompDetector(EventDetector):
    """Detects foot stomping (rapid downward foot movement)."""

    def __init__(self):
        super().__init__("stomp", cooldown=STOMP_COOLDOWN)
        self.velocity_threshold = STOMP_VELOCITY_THRESHOLD
        self.prev_foot_positions = None
        self.prev_time = None
        self.stability_frames = STOMP_STABILITY_FRAMES
        self.high_velocity_count = 0

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
            self.high_velocity_count += 1
        else:
            self.high_velocity_count = 0

        # Only trigger after seeing high velocity for multiple frames
        if self.high_velocity_count >= self.stability_frames and self.can_detect(frame_time):
            self.record_detection(frame_time)
            self.high_velocity_count = 0
            self.prev_foot_positions = current_positions
            self.prev_time = frame_time
            return DetectionEvent(
                name=self.name,
                timestamp=frame_time,
                confidence=min(1.0, max_velocity / (self.velocity_threshold * 2)),
                metadata={"foot": detected_foot, "velocity": max_velocity}
            )

        self.prev_foot_positions = current_positions
        self.prev_time = frame_time
        return None


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

class HandSectionDetector(EventDetector):
    """
    Detects when a hand is inside one of N horizontal screen sections.
    Waits for the hand to stay in that section for a short time before firing.
    """

    def __init__(self, hand: str, num_sections: int = 8, stability_time: float = 0.2):
        """
        :param hand: "left" or "right"
        :param num_sections: number of vertical sections from top to bottom
        :param stability_time: minimum time (s) hand must stay in section
        """
        self.hand = hand
        self.num_sections = num_sections
        self.stability_time = stability_time
        self.prev_section = None
        self.section_enter_time = None

        name = f"{hand}_section_detector"
        super().__init__(name, cooldown=0)  # cooldown handled by stability_time

        if hand == "left":
            self.wrist_id = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        else:
            self.wrist_id = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        if landmarks is None:
            self.prev_section = None
            self.section_enter_time = None
            return None

        wrist = landmarks[self.wrist_id]
        y_pos = wrist.y  # normalized 0 (top) → 1 (bottom)

        # Determine which section the hand is in
        section_height = 1.0 / self.num_sections
        section_index = min(int(y_pos / section_height), self.num_sections - 1)

        if self.prev_section != section_index:
            # Hand moved to a new section
            self.prev_section = section_index
            self.section_enter_time = frame_time
            return None
        else:
            # Hand is in same section
            time_in_section = frame_time - (self.section_enter_time or frame_time)
            if time_in_section >= self.stability_time:
                self.record_detection(frame_time)
                return DetectionEvent(
                    name=self.name,
                    timestamp=frame_time,
                    confidence=1.0,
                    metadata={
                        "hand": self.hand,
                        "section": section_index,
                        "time_in_section": time_in_section
                    }
                )
        return None




class PoseEstimator:
    """Main pose estimation and event detection system."""

    def __init__(self, detectors: List[EventDetector] = None):
        """
        Initialize pose estimator.

        Args:
            detectors: List of event detectors to use
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize detectors
        self.detectors = detectors or [
            ClapDetector(),
            StompDetector(),
            WristFlickDetector("left"),
            WristFlickDetector("right"),
            HandSectionDetector("left", num_sections=8, stability_time=0.2),
            HandSectionDetector("right", num_sections=8, stability_time=0.2)
        ]

        self.event_history: List[DetectionEvent] = []

    def add_detector(self, detector: EventDetector):
        """Add a new event detector."""
        self.detectors.append(detector)

    def process_frame(self, frame: np.ndarray, frame_time: float, debug: bool = False) -> Tuple[np.ndarray, List[DetectionEvent]]:
        """
        Process a single frame for pose estimation and event detection.

        Args:
            frame: Input image frame
            frame_time: Timestamp of the frame
            debug: Show debug info on frame

        Returns:
            Tuple of (annotated frame, list of detected events)
        """
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process with MediaPipe
        results = self.pose.process(image)

        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detected_events = []

        # Run event detection if pose landmarks are found
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Debug info collection
            debug_info = []

            # Get hand positions for debug
            left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
            hand_distance = np.sqrt(
                (left_wrist.x - right_wrist.x)**2 +
                (left_wrist.y - right_wrist.y)**2 +
                (left_wrist.z - right_wrist.z)**2
            )

            if debug:
                # Get ankle visibility for debug
                left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

                # Get detector info
                clap_detector = None
                left_flick = None
                right_flick = None
                for detector in self.detectors:
                    if isinstance(detector, ClapDetector):
                        clap_detector = detector
                    elif isinstance(detector, WristFlickDetector):
                        if detector.hand == "left":
                            left_flick = detector
                        else:
                            right_flick = detector

                # Clap detection debug
                if clap_detector:
                    state_colors = {
                        "close": "CLOSE",
                        "apart": "APART",
                        "medium": "MEDIUM",
                        "low_visibility": "LOW_VIS",
                        "cooldown": "COOLDOWN"
                    }
                    state_str = state_colors.get(clap_detector.current_state, "UNKNOWN")
                    debug_info.append(f"Clap: {state_str} | Close:{clap_detector.close_frame_count}/{CLAP_STABILITY_FRAMES} Apart:{clap_detector.apart_frame_count}/{CLAP_STABILITY_FRAMES}")
                    debug_info.append(f"Hand dist: {clap_detector.current_distance:.3f} (threshold: {CLAP_DISTANCE_THRESHOLD})")

                debug_info.append(f"Hand vis: L:{left_wrist.visibility:.2f} R:{right_wrist.visibility:.2f}")

                if left_flick:
                    debug_info.append(f"L Wrist: spd={left_flick.current_speed:.2f} dir={left_flick.current_angular_velocity:.2f} [{left_flick.flick_frame_count}/{FLICK_STABILITY_FRAMES}]")
                if right_flick:
                    debug_info.append(f"R Wrist: spd={right_flick.current_speed:.2f} dir={right_flick.current_angular_velocity:.2f} [{right_flick.flick_frame_count}/{FLICK_STABILITY_FRAMES}]")

                # Also print to console occasionally
                if hasattr(self, '_last_debug_print'):
                    if frame_time - self._last_debug_print > 0.5:  # Print every 0.5s
                        clap_info = ""
                        if clap_detector:
                            clap_info = f"Clap: {clap_detector.current_state} Close:{clap_detector.close_frame_count}/{CLAP_STABILITY_FRAMES}"

                        flick_info = ""
                        if left_flick:
                            flick_info += f" | L_Flick: S:{left_flick.current_speed:.2f} D:{left_flick.current_angular_velocity:.2f} [{left_flick.flick_frame_count}/{FLICK_STABILITY_FRAMES}]"
                        if right_flick:
                            flick_info += f" | R_Flick: S:{right_flick.current_speed:.2f} D:{right_flick.current_angular_velocity:.2f} [{right_flick.flick_frame_count}/{FLICK_STABILITY_FRAMES}]"

                        print(f"[DEBUG] {clap_info}{flick_info}")
                        self._last_debug_print = frame_time
                else:
                    self._last_debug_print = frame_time

            for detector in self.detectors:
                event = detector.detect(landmarks, frame_time)
                if event:
                    detected_events.append(event)
                    self.event_history.append(event)
                    print(f"[{event.name.upper()}] Detected at {event.timestamp:.2f}s "
                          f"(confidence: {event.confidence:.2f})")

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2
                )
            )

            # Draw line between hands for clap detection debug
            if debug:
                h, w, _ = image.shape
                left_x, left_y = int(left_wrist.x * w), int(left_wrist.y * h)
                right_x, right_y = int(right_wrist.x * w), int(right_wrist.y * h)

                # Get clap detector for visualization
                clap_det = None
                for det in self.detectors:
                    if isinstance(det, ClapDetector):
                        clap_det = det
                        break

                # Line color: green ONLY when clap just detected, otherwise white
                if clap_det and clap_det.just_detected:
                    color = (0, 255, 0)  # Bright green on detection
                    thickness = 4
                else:
                    color = (255, 255, 255)  # White normally
                    thickness = 1

                cv2.line(image, (left_x, left_y), (right_x, right_y), color, thickness)

                # Draw wrist flick debug indicators
                for detector in self.detectors:
                    if isinstance(detector, WristFlickDetector):
                        if detector.hand == "left":
                            wrist_x, wrist_y = left_x, left_y
                        else:
                            wrist_x, wrist_y = right_x, right_y

                        # Only show green circle when flick just detected
                        if detector.just_detected:
                            # Bright green flash on detection
                            radius = 30
                            circle_color = (0, 255, 0)
                            cv2.circle(image, (wrist_x, wrist_y), radius, circle_color, 4)

                            # Show which hand
                            hand_label = "R" if detector.hand == "right" else "L"
                            offset_x = 40 if detector.hand == "right" else -70
                            text_x = wrist_x + offset_x
                            text_y = wrist_y
                            cv2.putText(image, f"{hand_label}-FLICK!", (text_x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw section detection visual cues
                left_section_det = None
                right_section_det = None
                for detector in self.detectors:
                    if isinstance(detector, HandSectionDetector):
                        if detector.hand == "left":
                            left_section_det = detector
                        else:
                            right_section_det = detector

                # Draw section boundaries and highlights
                if left_section_det or right_section_det:
                    # Use the first detector's num_sections (should be same for both)
                    num_sections = left_section_det.num_sections if left_section_det else right_section_det.num_sections
                    section_height = h / num_sections

                    # Draw horizontal lines for section boundaries (subtle gray)
                    for i in range(1, num_sections):
                        y = int(i * section_height)
                        cv2.line(image, (0, y), (w, y), (100, 100, 100), 1)

                    # Highlight left hand's section (appears on RIGHT side of screen due to mirror)
                    if left_section_det and left_section_det.prev_section is not None:
                        section_idx = left_section_det.prev_section
                        y_top = int(section_idx * section_height)
                        y_bottom = int((section_idx + 1) * section_height)

                        # Draw colored overlay on right half (left hand appears on right due to mirror)
                        overlay = image.copy()
                        cv2.rectangle(overlay, (w // 2, y_top), (w, y_bottom), (255, 0, 0), -1)  # Blue for left
                        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

                        # Add section number
                        cv2.putText(image, f"L-{section_idx}", (w - 100, y_top + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Highlight right hand's section (appears on LEFT side of screen due to mirror)
                    if right_section_det and right_section_det.prev_section is not None:
                        section_idx = right_section_det.prev_section
                        y_top = int(section_idx * section_height)
                        y_bottom = int((section_idx + 1) * section_height)

                        # Draw colored overlay on left half (right hand appears on left due to mirror)
                        overlay = image.copy()
                        cv2.rectangle(overlay, (0, y_top), (w // 2, y_bottom), (0, 255, 0), -1)  # Green for right
                        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

                        # Add section number
                        cv2.putText(image, f"R-{section_idx}", (10, y_top + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            if debug:
                debug_info = ["No pose detected"]

        # Display recent events on frame
        y_offset = 30
        for event in self.event_history[-5:]:  # Show last 5 events
            cv2.putText(
                image,
                f"{event.name}: {event.confidence:.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_offset += 30

        # Display debug info
        if debug and 'debug_info' in locals():
            y_offset = image.shape[0] - 20
            for info in debug_info:
                cv2.putText(
                    image,
                    info,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
                y_offset -= 25

        return image, detected_events

    def run(self, source=0, show_window: bool = False, debug: bool = False, output_video: str = None):
        """
        Run the pose estimator with live camera feed or video file.

        Args:
            source: Camera device index (int) or video file path (str)
            show_window: Whether to show OpenCV window (requires display)
            debug: Show debug visualizations on frames
            output_video: Path to save annotated output video (optional)
        """
        print("=" * 50)
        print("Starting pose estimation and event detection...")
        print("=" * 50)
        print("\nDetectors enabled:")
        for detector in self.detectors:
            print(f"  - {detector.name}")
        print()

        # Try to open camera or video file
        if isinstance(source, str):
            print(f"Opening video file: {source}...")
        else:
            print(f"Opening camera {source}...")

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            if isinstance(source, str):
                print(f"ERROR: Could not open video file: {source}")
                print("\nTroubleshooting:")
                print("  - Check if the file exists")
                print("  - Check if the file format is supported (mp4, avi, mov, etc.)")
            else:
                print(f"ERROR: Could not open camera {source}")
                print("\nTroubleshooting:")
                print("  - Check if camera is connected")
                print("  - Try a different camera index (0, 1, 2, etc.)")
                print("  - On WSL2, use a video file instead: python main.py video.mp4")
                print("  - On Mac, grant camera permissions in System Preferences")
            return

        # Test read
        success, test_frame = cap.read()
        if not success:
            print("ERROR: Could not read first frame")
            cap.release()
            return

        source_type = "Video file" if isinstance(source, str) else "Camera"
        print(f"✓ {source_type} opened successfully")
        print(f"✓ Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")

        # Get total frames if video file
        video_fps = 30.0  # Default FPS
        if isinstance(source, str):
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps if video_fps > 0 else 0
            print(f"✓ Total frames: {total_frames}")
            print(f"✓ FPS: {video_fps:.1f}")
            print(f"✓ Duration: {duration:.1f}s")

        # Setup video writer if output requested
        video_writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (test_frame.shape[1], test_frame.shape[0])
            video_writer = cv2.VideoWriter(output_video, fourcc, video_fps, frame_size)
            print(f"✓ Will save output to: {output_video}")

        if debug:
            print("✓ Debug mode enabled")

        print(f"\nProcessing frames... Press Ctrl+C to stop\n")

        start_time = time.time()
        frame_count = 0

        is_video_file = isinstance(source, str)

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    if is_video_file:
                        print("\nEnd of video reached.")
                        break
                    else:
                        print("Warning: Failed to read frame, retrying...")
                        continue

                frame_time = time.time() - start_time
                frame_count += 1

                # Process frame
                annotated_frame, events = self.process_frame(frame, frame_time, debug=debug)

                # Write to output video if requested
                if video_writer:
                    video_writer.write(annotated_frame)

                # Show FPS every 30 frames
                if frame_count % 30 == 0:
                    fps = frame_count / frame_time
                    print(f"[INFO] Running at {fps:.1f} FPS | Events detected: {len(self.event_history)}")

                # Display window if requested (won't work in WSL2 without X server)
                if show_window:
                    cv2.imshow('Pose Estimation & Event Detection', annotated_frame)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break
                elif is_video_file:
                    # Small delay for video files to prevent processing too fast
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
                print(f"\n✓ Output video saved to: {output_video}")
            if show_window:
                cv2.destroyAllWindows()
            self.pose.close()

            # Print summary
            print("\n" + "=" * 50)
            print("Detection Summary")
            print("=" * 50)
            print(f"Total runtime: {time.time() - start_time:.1f}s")
            print(f"Total frames: {frame_count}")
            print(f"Total events detected: {len(self.event_history)}")

            if self.event_history:
                print("\nEvent breakdown:")
                event_counts = {}
                for event in self.event_history:
                    event_counts[event.name] = event_counts.get(event.name, 0) + 1
                for event_name, count in event_counts.items():
                    print(f"  {event_name}: {count}")
            else:
                print("\nNo events detected.")


def main():
    """Main entry point for testing."""
    # Parse simple command-line args
    source = 0  # Default to camera 0
    show_window = False
    debug = False
    output_video = None

    if len(sys.argv) > 1 and sys.argv[1] not in ['--show', '--debug', '--output']:
        # First arg is either camera index or video file path
        try:
            source = int(sys.argv[1])  # Try as camera index
        except ValueError:
            source = sys.argv[1]  # Treat as file path

    if '--show' in sys.argv:
        show_window = True

    if '--debug' in sys.argv:
        debug = True

    # Check for --output flag
    if '--output' in sys.argv:
        try:
            output_idx = sys.argv.index('--output')
            if output_idx + 1 < len(sys.argv):
                output_video = sys.argv[output_idx + 1]
            else:
                # Auto-generate output filename
                if isinstance(source, str):
                    output_video = source.replace('.', '_output.')
                else:
                    output_video = 'output.mp4'
        except (ValueError, IndexError):
            output_video = 'output.mp4'

    if len(sys.argv) == 1:
        print("Usage:")
        print(f"  {sys.argv[0]}                              # Use camera 0")
        print(f"  {sys.argv[0]} 1                            # Use camera 1")
        print(f"  {sys.argv[0]} video.mp4                    # Use video file")
        print(f"  {sys.argv[0]} video.mp4 --debug            # Show debug info")
        print(f"  {sys.argv[0]} video.mp4 --output out.mp4   # Save output video")
        print(f"  {sys.argv[0]} video.mp4 --debug --output   # Debug + save")
        print()
        print("Running with default camera 0...")
        print()

    # Create pose estimator with default detectors
    estimator = PoseEstimator()

    # Example: Add custom detectors
    # estimator.add_detector(YourCustomDetector())

    # Run the system
    estimator.run(source=source, show_window=show_window, debug=debug, output_video=output_video)


if __name__ == "__main__":
    main()
