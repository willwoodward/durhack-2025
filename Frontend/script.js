const video = document.getElementById("live-stream");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
const instrument_html = document.getElementById("instrument");
const note_html = document.getElementById("note");
const bpm_html = document.getElementById("bpm");
const video_container = document.getElementById("video-container");
const topLeftBox = document.getElementById("top-left-box");
const topRightBox = document.getElementById("top-right-box");
const stop_button = document.getElementById("stop_button");
const effect_button = document.getElementById("effect_button");
const legacy_button = document.getElementById("legacy_button");

// Connect to WebSocket server
const ws = new WebSocket("ws://localhost:3000");

ws.onopen = () => {
  console.log("WebSocket connection opened");
};

ws.onerror = (err) => {
  console.error("WebSocket error:", err);
};

const collider = new ClapCollider();
collider.start();

legacy_button.onclick = () => {
  const isLegacy = video_container.classList.contains("legacy");

  if (isLegacy) {
    video_container.classList.remove("legacy");
    video_container.classList.add("normal");
  } else {
    video_container.classList.remove("normal");
    video_container.classList.add("legacy");
  }
};

effect_button.onclick = () => {
  const pulseDiv = document.createElement("div");
  pulseDiv.classList.add("pulse");
  document.body.appendChild(pulseDiv);

  setTimeout(() => {
    pulseDiv.remove();
  }, 600);
};

stop_button.onclick = () => collider.stop();

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.event_name === "right_hand_upper") {
    topLeftBox.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
    topRightBox.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
  } else if (data.event_name === "left_hand_upper") {
    topLeftBox.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
    topRightBox.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
  } else {
    topLeftBox.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
    topRightBox.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
  }

  if (data.event_name === "clap") {
    const pulseDiv = document.createElement("div");
    pulseDiv.classList.add("pulse");
    document.body.appendChild(pulseDiv);

    setTimeout(() => {
      pulseDiv.remove();
    }, 600);
  }

  console.log(`🎵 ${data.event_name.toUpperCase()} detected!`, {
    event: data.event_name,
    onset_time: new Date(data.onset_time * 1000).toISOString(),
    offset_time: new Date(data.offset_time * 1000).toISOString(),
    instrument: data.instrument,
    note: data.note,
    bpm: data.bpm,
    metadata: data.metadata,
  });

  collider.handleEvent(
    data.event_name,
    data.onset_time,
    data.offset_time,
    data.metadata
  );

  instrument_html.textContent = data.instrument;
  note_html.textContent = "Note: " + data.note;
  bpm_html.textContent = "BPM: " + data.bpm;
  if (data.event_name == "right_swipe_right_to_left") {
    toggleNoteOverlay(true);
  } else if (data.event_name == "left_swipe_left_to_right") {
    toggleNoteOverlay(false);
  }
};

ws.onclose = () => {
  console.log("WebSocket closed");
};

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;

      video.addEventListener("loadedmetadata", () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        sendFrames();
      });

      function sendFrames() {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(
          (blob) => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(blob);
            }
          },
          "image/jpeg",
          1
        );

        setTimeout(sendFrames, 30);
      }
    })
    .catch((error) => {
      console.error("Error accessing camera:", error);
    });
} else {
  console.log("getUserMedia is not supported in this browser.");
}

// Add note overlay lines
const numLines = 7;
const overlayLines = [];
for (let i = 0; i < numLines; i++) {
  const line = document.createElement("div");
  line.className = "overlay-line";
  line.style.top = `${((i + 1) / (numLines + 1)) * 100}vh`;
  line.style.display = "none"; // Initially hidden (start in drum mode)
  video_container.appendChild(line);
  overlayLines.push(line);
}

// Track whether we're in note mode (synths) or drum mode
let noteMode = false;

// Toggle between note mode (synths) and drum mode
function toggleNoteOverlay(show) {
  noteMode = show;
  collider.setNoteMode(show);

  overlayLines.forEach((line) => {
    line.style.display = show ? "block" : "none";
  });
  topLeftBox.style.display = show ? "none" : "block";
  topRightBox.style.display = show ? "none" : "block";

  console.log(`🎼 Mode switched to: ${show ? "SYNTHS/CHORDS" : "DRUMS"}`);
}
toggleNoteOverlay(true);
