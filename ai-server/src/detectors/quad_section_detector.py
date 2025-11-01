from typing import Optional
import mediapipe as mp

from . import EventDetector, DetectionEvent

class QuadSectionDetector(EventDetector):
    """
    Detects when a hand enters the top-left or top-right quadrant.
    Fires once per quadrant entry (only when entering top section, section 0).
    """

    def __init__(self, hand: str, num_sections: int = 2, stability_time: float = 0.05):
        """
        :param hand: "left" or "right"
        :param num_sections: number of vertical sections from top to bottom (2 for top/bottom)
        :param stability_time: minimum time (s) hand must stay in section before firing
        """
        self.hand = hand
        self.num_sections = num_sections
        self.stability_time = stability_time
        self.prev_section = None
        self.section_enter_time = None
        self.fired_for_current_section = False  # Track if we already fired for this section entry

        name = f"{hand}_quad_detector"
        super().__init__(name, cooldown=0.3)  # Add cooldown to prevent spam

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
        y_pos = wrist.y

        # Determine which section the hand is in
        section_height = 1.0 / self.num_sections
        section_index = min(int(y_pos / section_height), self.num_sections - 1)

        # Only detect top section (section 0)
        if section_index != 0:
            # Hand not in top quadrant, reset state
            if self.prev_section == 0:
                # Hand left top quadrant
                self.fired_for_current_section = False
            self.prev_section = section_index
            self.section_enter_time = None
            return None

        # Hand is in top quadrant (section 0)
        if self.prev_section != section_index:
            # Hand just entered top quadrant
            self.prev_section = section_index
            self.section_enter_time = frame_time
            self.fired_for_current_section = False
            return None
        else:
            # Hand is still in top quadrant
            time_in_section = frame_time - (self.section_enter_time or frame_time)

            # Fire once when stability time is reached and we haven't fired yet
            if time_in_section >= self.stability_time and not self.fired_for_current_section:
                if self.can_detect(frame_time):
                    self.fired_for_current_section = True
                    self.record_detection(frame_time)
                    return DetectionEvent(
                        name="left_hand_upper" if self.hand == "left" else "right_hand_upper",
                        timestamp=frame_time,
                        confidence=1.0,
                        onset_time=self.section_enter_time or frame_time,
                        offset_time=frame_time,
                        metadata={
                            "hand": self.hand,
                            "section": section_index,
                            "quadrant": "top-left" if self.hand == "left" else "top-right",
                            "time_in_section": time_in_section
                        }
                    )
        return None
