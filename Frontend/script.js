const video = document.getElementById("live-stream");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
const instrument_html = document.getElementById("instrument");
const note_html = document.getElementById("note");
const bpm_html = document.getElementById("bpm");
const video_container = document.getElementById("video-container");
const topLeftBox = document.getElementById("top-left-box");
const topRightBox = document.getElementById("top-right-box");

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

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.event_name === "left_hand_upper") {
    //Make the upper left quadrant less transparent
    topLeftBox.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
    topRightBox.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
  } else if (data.event_name === "right_hand_upper") {
    topLeftBox.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
    topRightBox.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
  } else {
    topLeftBox.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
    topRightBox.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
  }

  // Log all events
  console.log(`🎵 ${data.event_name.toUpperCase()} detected!`, {
    event: data.event_name,
    onset_time: new Date(data.onset_time * 1000).toISOString(),
    offset_time: new Date(data.offset_time * 1000).toISOString(),
    instrument: data.instrument,
    note: data.note,
    bpm: data.bpm,
  });

  collider.handleEvent(data.event_name, data.onset_time, data.offset_time);

  instrument_html.textContent = data.instrument;
  note_html.textContent = "Note: " + data.note;
  bpm_html.textContent = "BPM: " + data.bpm;
  if (data.instrument == "Piano") {
    toggleNoteOverlay(true);
  } else {
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

      // Start sending frames after video metadata is loaded
      video.addEventListener("loadedmetadata", () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        sendFrames();
      });

      function sendFrames() {
        // Draw current frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to Blob and send over WebSocket
        canvas.toBlob(
          (blob) => {
            if (ws.readyState === WebSocket.OPEN) {
              // Send Blob directly (browser will handle binary)
              ws.send(blob);
              //console.log("Frame sent, size:", blob.size);
            }
          },
          "image/jpeg",
          1
        );

        // Schedule next frame (~10 fps)
        setTimeout(sendFrames, 30);
      }
    })
    .catch((error) => {
      console.error("Error accessing camera:", error);
    });
} else {
  console.log("getUserMedia is not supported in this browser.");
}

// add note overlay

const numLines = 7;
const overlayLines = [];
for (let i = 0; i < numLines; i++) {
  const line = document.createElement("div");
  line.className = "overlay-line";
  line.style.top = `${((i + 1) / (numLines + 1)) * 100}vh`;
  video_container.appendChild(line);
  overlayLines.push(line);
}

// Function to toggle overlay
function toggleNoteOverlay(show) {
  // show is expected to be a boolean; set display accordingly
  overlayLines.forEach((line) => {
    line.style.display = show ? "block" : "none";
  });
}

let note_overlay = true;

// show after 3 seconds (demo)
// setTimeout(() => toggleNoteOverlay(false), 1000);
