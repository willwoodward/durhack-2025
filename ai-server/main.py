#!/usr/bin/env python3
"""
Pose estimation and event detection system using MediaPipe.
Detects events like hand clapping, foot stomping, and wrist flicks.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple
import time
import sys
import asyncio
import websockets
import json
import threading

from src.detectors import DetectionEvent, EventDetector, ClapDetector, StompDetector, SwipeDetector, WristFlickDetector, HandSectionDetector, QuadSectionDetector


class PoseEstimator:
    """Main pose estimation and event detection system."""

    def __init__(self, detectors: List[EventDetector] = None, event_callback=None):
        """
        Initialize pose estimator.

        Args:
            detectors: List of event detectors to use
            event_callback: Optional callback function to call when events are detected
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
            HandSectionDetector("right", num_sections=8, stability_time=0.2),
            QuadSectionDetector("left", num_sections=2, stability_time=0.1),
            QuadSectionDetector("right", num_sections=2, stability_time=0.1),
            SwipeDetector("left", "left_to_right")
        ]

        self.event_history: List[DetectionEvent] = []
        self.event_callback = event_callback
        self.start_unix_time = None  # Unix timestamp when processing starts

    def add_detector(self, detector: EventDetector):
        """Add a new event detector."""
        self.detectors.append(detector)

    def process_frame(self, frame: np.ndarray, frame_time: float, debug: bool = False) -> Tuple[np.ndarray, List[DetectionEvent]]:
        """
        Process a single frame for pose estimation and event detection.

        Args:
            frame: Input image frame
            frame_time: Unix timestamp of the frame
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

                if left_flick:
                    debug_info.append(f"L Wrist: spd={left_flick.current_speed:.2f} dir={left_flick.current_angular_velocity:.2f}")
                if right_flick:
                    debug_info.append(f"R Wrist: spd={right_flick.current_speed:.2f} dir={right_flick.current_angular_velocity:.2f}")

                # Also print to console occasionally
                if hasattr(self, '_last_debug_print'):
                    if frame_time - self._last_debug_print > 0.5:  # Print every 0.5s
                        self._last_debug_print = frame_time
                else:
                    self._last_debug_print = frame_time

            for detector in self.detectors:
                event = detector.detect(landmarks, frame_time)
                if event:
                    detected_events.append(event)
                    self.event_history.append(event)

                    # Call event callback if provided
                    if self.event_callback:
                        self.event_callback(event)

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

                # Draw swipe detection visual cues
                for detector in self.detectors:
                    if isinstance(detector, SwipeDetector):
                        if detector.hand == "left":
                            wrist_x, wrist_y = left_x, left_y
                        else:
                            wrist_x, wrist_y = right_x, right_y

                        # Show arrow when swipe is detected
                        if detector.just_detected:
                            arrow_length = 80
                            arrow_color = (0, 255, 255)  # Yellow/cyan

                            # Draw arrow in swipe direction
                            if detector.direction == "left_to_right":
                                start_pt = (wrist_x - arrow_length // 2, wrist_y)
                                end_pt = (wrist_x + arrow_length // 2, wrist_y)
                            else:  # right_to_left
                                start_pt = (wrist_x + arrow_length // 2, wrist_y)
                                end_pt = (wrist_x - arrow_length // 2, wrist_y)

                            cv2.arrowedLine(image, start_pt, end_pt, arrow_color, 4, tipLength=0.3)

                            # Label
                            direction_label = "→" if detector.direction == "left_to_right" else "←"
                            hand_label = "R" if detector.hand == "right" else "L"
                            cv2.putText(image, f"{hand_label}-SWIPE {direction_label}",
                                       (wrist_x - 60, wrist_y - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, arrow_color, 2)

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

        print("\nProcessing frames... Press Ctrl+C to stop\n")

        start_time = time.time()
        self.start_unix_time = start_time  # Store Unix start time
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

                # Use absolute Unix timestamp instead of relative time
                current_unix_time = time.time()
                frame_time = current_unix_time - start_time  # Relative time for FPS calculation
                frame_count += 1

                # Process frame with absolute Unix timestamp
                annotated_frame, events = self.process_frame(frame, current_unix_time, debug=debug)

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


# ==================== WEBSOCKET SERVER ====================

# Global set of connected clients
connected_clients = set()

async def websocket_handler(websocket):
    """Handle WebSocket connections and process video frames from frontend."""
    connected_clients.add(websocket)
    print(f"[WEBSOCKET] Frontend connected from {websocket.remote_address}. Total clients: {len(connected_clients)}")

    # Create a pose estimator instance for this client
    estimator = PoseEstimator()
    estimator.start_unix_time = time.time()

    try:
        async for message in websocket:
            try:
                # Frontend sends binary blob (JPEG image)
                if isinstance(message, bytes):
                    # Decode binary JPEG frame
                    nparr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Process frame with pose estimation
                        current_unix_time = time.time()
                        _, events = estimator.process_frame(frame, current_unix_time, debug=False)

                        # Send events back to frontend
                        for event in events:
                            # Allow specific events or any section/quad detector events
                            allowed_events = ["clap", "stomp", "left_hand_upper", "right_hand_upper"]
                            is_section_event = "section" in event.name or "quad" in event.name

                            if event.name in allowed_events or is_section_event:
                                # Send in format frontend expects
                                event_payload = {
                                    "instrument": "Drums",
                                    "note": event.name.upper(),
                                    "bpm": 120,
                                    "event_name": event.name,
                                    "onset_time": event.onset_time,
                                    "offset_time": event.offset_time,
                                    "metadata": event.metadata
                                }
                                await websocket.send(json.dumps(event_payload))
                                print(f"[WEBSOCKET] Sent {event.name} to frontend: onset={event.onset_time:.2f}, offset={event.offset_time:.2f}")
                    else:
                        print("[WEBSOCKET] Failed to decode frame")
                else:
                    # Handle JSON messages if needed
                    try:
                        data = json.loads(message)
                        print(f"[WEBSOCKET] Received JSON: {data}")
                    except:
                        print(f"[WEBSOCKET] Unknown message type")

            except Exception as e:
                print(f"[WEBSOCKET] Error processing frame: {e}")
                import traceback
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosed:
        print("[WEBSOCKET] Connection closed")
    finally:
        connected_clients.remove(websocket)
        print(f"[WEBSOCKET] Frontend disconnected. Total clients: {len(connected_clients)}")


# Global event loop for websocket server
websocket_loop = None

async def start_websocket_server(host='0.0.0.0', port=3000):
    """Start the WebSocket server."""
    async with websockets.serve(websocket_handler, host, port):
        print(f"[WEBSOCKET] Server started on ws://{host}:{port}")
        await asyncio.Future()  # Run forever


def run_websocket_server(host='0.0.0.0', port=3000):
    """Run WebSocket server in a separate thread."""
    global websocket_loop
    websocket_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(websocket_loop)
    websocket_loop.run_until_complete(start_websocket_server(host, port))


# ======================================================


def main():
    """Main entry point."""
    # Parse simple command-line args
    source = None  # No video source by default
    show_window = False
    debug = False
    output_video = None
    run_server = False

    if len(sys.argv) > 1 and sys.argv[1] not in ['--show', '--debug', '--output', '--server']:
        # First arg is either camera index or video file path
        try:
            source = int(sys.argv[1])  # Try as camera index
        except ValueError:
            source = sys.argv[1]  # Treat as file path

    if '--show' in sys.argv:
        show_window = True

    if '--debug' in sys.argv:
        debug = True

    if '--server' in sys.argv:
        run_server = True

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
        print(f"  {sys.argv[0]} --server                          # Run WebSocket server for frontend")
        print(f"  {sys.argv[0]} video.mp4                         # Debug: Process video file")
        print(f"  {sys.argv[0]} video.mp4 --debug                 # Debug: Show debug visualizations")
        print(f"  {sys.argv[0]} video.mp4 --server --debug        # Debug: Video file + WebSocket server")
        print(f"  {sys.argv[0]} 0                                 # Debug: Use camera 0")
        print()
        print("Running WebSocket server only (production mode)...")
        run_server = True

    # Start WebSocket server if requested
    if run_server:
        print("=" * 50)
        print("Starting AI Pose Estimation WebSocket Server")
        print("=" * 50)
        server_thread = threading.Thread(target=run_websocket_server, daemon=True)
        server_thread.start()
        time.sleep(1)  # Give server time to start
        print("✓ WebSocket server started on ws://0.0.0.0:3000")
        print("✓ Frontend should connect to: ws://localhost:3000")
        print("✓ Receives: Binary JPEG frames")
        print("✓ Sends: JSON events with {instrument, note, bpm, event_name, onset_time, offset_time}")
        print()

    # Run debug video processing if source is provided
    if source is not None:
        print("Running debug video processor...")
        estimator = PoseEstimator()
        estimator.run(source=source, show_window=show_window, debug=debug, output_video=output_video)
    elif run_server:
        # Server-only mode: keep running
        print("Server running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        print("Error: Please specify --server or provide a video source")
        sys.exit(1)


if __name__ == "__main__":
    main()
