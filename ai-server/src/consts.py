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

# Swipe Detection (horizontal hand motion)
SWIPE_VELOCITY_THRESHOLD = 0.3      # Min horizontal velocity to detect swipe
SWIPE_MIN_DISTANCE = 0.15           # Min total distance traveled to confirm swipe
SWIPE_STABILITY_FRAMES = 3          # Consecutive frames moving in correct direction
SWIPE_COOLDOWN = 0.5                # Seconds between swipe detections

# Chop Detection (vertical hand motion)
CHOP_VELOCITY_THRESHOLD = 0.5       # Min vertical distance between wrist and elbow to detect chop
CHOP_STABILITY_FRAMES = 3           # Consective frames needed to confirm a chop
CHOP_COOLDOWN = 0.5                 # Minimum seconds between chop detections



# Visibility Thresholds                # Min landmark visibility to detect events (0-1)
MIN_VISIBILITY = 0.5
