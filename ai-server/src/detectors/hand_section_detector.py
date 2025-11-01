from typing import Optional
import mediapipe as mp

from . import EventDetector, DetectionEvent

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
        self.fired_for_current_section = False  # Prevent spam

        name = f"{hand}_section_detector"
        super().__init__(name, cooldown=0.2)  # Add cooldown

        if hand == "left":
            self.wrist_id = mp.solutions.pose.PoseLandmark.LEFT_WRIST.value
        else:
            self.wrist_id = mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value

    def detect(self, landmarks, frame_time: float) -> Optional[DetectionEvent]:
        if landmarks is None:
            self.prev_section = None
            self.section_enter_time = None
            self.fired_for_current_section = False
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
            self.fired_for_current_section = False  # Reset on section change
            return None
        else:
            # Hand is in same section
            time_in_section = frame_time - (self.section_enter_time or frame_time)

            # Fire once when stability time is reached and we haven't fired yet
            if time_in_section >= self.stability_time and not self.fired_for_current_section:
                if self.can_detect(frame_time):
                    self.fired_for_current_section = True
                    self.record_detection(frame_time)
                    return DetectionEvent(
                        name=self.name,
                        timestamp=frame_time,
                        confidence=1.0,
                        onset_time=self.section_enter_time or frame_time,
                        offset_time=frame_time,
                        metadata={
                            "hand": self.hand,
                            "section": section_index,
                            "time_in_section": time_in_section
                        }
                    )
        return None
