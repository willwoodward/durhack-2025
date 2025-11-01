#!/usr/bin/env python3
"""
Pose estimation and event detection system using MediaPipe.
Detects events like hand clapping, foot stomping, and hand hits.
"""

import cv2
import mediapipe as mp
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
import time
import sys


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


class ClapDetector(EventDetector):
    """Detects hand clapping gestures."""

    def __init__(self, distance_threshold: float = 0.1):
        super().__init__("clap", cooldown=0.3)
        self.distance_threshold = distance_threshold
        self.was_apart = True  # Track if hands were apart

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        if not self.can_detect(frame_time):
            return None

        # Get hand landmarks
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]

        # Calculate distance between hands
        distance = np.sqrt(
            (left_wrist.x - right_wrist.x)**2 +
            (left_wrist.y - right_wrist.y)**2 +
            (left_wrist.z - right_wrist.z)**2
        )

        # Detect clap: hands were apart and are now together
        if distance < self.distance_threshold and self.was_apart:
            self.was_apart = False
            self.record_detection(frame_time)
            return DetectionEvent(
                name=self.name,
                timestamp=frame_time,
                confidence=1.0 - (distance / self.distance_threshold),
                metadata={"distance": distance}
            )
        elif distance > self.distance_threshold * 1.5:
            self.was_apart = True

        return None


class StompDetector(EventDetector):
    """Detects foot stomping (rapid downward foot movement)."""

    def __init__(self, velocity_threshold: float = 0.15):
        super().__init__("stomp", cooldown=0.4)
        self.velocity_threshold = velocity_threshold
        self.prev_foot_positions = None
        self.prev_time = None

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        # Get foot landmarks
        left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]

        current_positions = {
            'left': np.array([left_ankle.x, left_ankle.y, left_ankle.z]),
            'right': np.array([right_ankle.x, right_ankle.y, right_ankle.z])
        }

        if self.prev_foot_positions is not None and self.prev_time is not None:
            dt = frame_time - self.prev_time
            if dt > 0:
                # Check each foot for stomp
                for foot_name, current_pos in current_positions.items():
                    prev_pos = self.prev_foot_positions[foot_name]

                    # Calculate vertical velocity (y-axis, positive is downward)
                    vertical_velocity = (current_pos[1] - prev_pos[1]) / dt

                    # Detect stomp: rapid downward movement followed by stop
                    if vertical_velocity > self.velocity_threshold and self.can_detect(frame_time):
                        self.record_detection(frame_time)
                        self.prev_foot_positions = current_positions
                        self.prev_time = frame_time
                        return DetectionEvent(
                            name=self.name,
                            timestamp=frame_time,
                            confidence=min(1.0, vertical_velocity / (self.velocity_threshold * 2)),
                            metadata={"foot": foot_name, "velocity": vertical_velocity}
                        )

        self.prev_foot_positions = current_positions
        self.prev_time = frame_time
        return None


class RightHandHitDetector(EventDetector):
    """Detects when right hand makes a hitting motion (rapid forward movement)."""

    def __init__(self, velocity_threshold: float = 0.3):
        super().__init__("right_hand_hit", cooldown=0.4)
        self.velocity_threshold = velocity_threshold
        self.prev_position = None
        self.prev_time = None

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        # Get right hand landmark
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
        current_position = np.array([right_wrist.x, right_wrist.y, right_wrist.z])

        if self.prev_position is not None and self.prev_time is not None:
            dt = frame_time - self.prev_time
            if dt > 0 and self.can_detect(frame_time):
                # Calculate velocity
                displacement = current_position - self.prev_position
                velocity = np.linalg.norm(displacement) / dt

                # Check if it's a forward motion (negative z-direction in MediaPipe)
                forward_motion = displacement[2] < -0.02

                if velocity > self.velocity_threshold and forward_motion:
                    self.record_detection(frame_time)
                    self.prev_position = current_position
                    self.prev_time = frame_time
                    return DetectionEvent(
                        name=self.name,
                        timestamp=frame_time,
                        confidence=min(1.0, velocity / (self.velocity_threshold * 2)),
                        metadata={"velocity": velocity}
                    )

        self.prev_position = current_position
        self.prev_time = frame_time
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
            RightHandHitDetector()
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
                debug_info.append(f"Hand dist: {hand_distance:.3f}")
                # Also print to console occasionally
                if hasattr(self, '_last_debug_print'):
                    if frame_time - self._last_debug_print > 0.5:  # Print every 0.5s
                        print(f"[DEBUG] Hand distance: {hand_distance:.3f} (threshold: 0.1)")
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

                # Color based on distance (green = close, red = far)
                color = (0, 255, 0) if hand_distance < 0.1 else (0, 165, 255) if hand_distance < 0.15 else (0, 0, 255)
                cv2.line(image, (left_x, left_y), (right_x, right_y), color, 2)
                cv2.circle(image, (left_x, left_y), 5, (255, 0, 0), -1)
                cv2.circle(image, (right_x, right_y), 5, (0, 0, 255), -1)

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
