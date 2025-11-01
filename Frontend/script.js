const video = document.getElementById("live-stream");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
const instrument_html = document.getElementById("instrument");
const note_html = document.getElementById("note");
const bpm_html = document.getElementById("bpm");

// Connect to WebSocket server
const ws = new WebSocket("ws://localhost:3000");

ws.onopen = () => {
  console.log("WebSocket connection opened");
};

ws.onerror = (err) => {
  console.error("WebSocket error:", err);
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.instrument === "Piano") {
    instrument_html.innerHTML = `Piano 
      <img src="../Icons/piano.png" alt="Piano" 
           style="width: 60px; height: 40px; vertical-align: middle; margin-left: 0px;">`;
  } else if (data.instrument === "Kickdrum") {
    instrument_html.innerHTML = `Kick Drum 
      <img src="../Icons/kick.png" alt="Kick Drum" 
           style="width: 40px; height: 40px; vertical-align: middle; margin-left: 0px;">`;
  } else if (data.instrument === "Snare") {
    instrument_html.innerHTML = `Snare 
      <img src="../Icons/snare.png" alt="Snare" 
           style="width: 40px; height: 40px; vertical-align: middle; margin-left: 0px;">`;
  } else if (data.instrument === "High Hat"){
    instrument_html.innerHTML = `High Hat 
      <img src="../Icons/highhat.png" alt="High Hat" 
           style="width: 40px; height: 40px; vertical-align: middle; margin-left: 0px;">`;
  }else{
    instrument_html.textContent = data.instrument;
  }

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
          0.7
        );

        // Schedule next frame (~10 fps)
        setTimeout(sendFrames, 100);
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
  document.body.appendChild(line);
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
