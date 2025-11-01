const video = document.getElementById("live-stream");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

// Connect to WebSocket server
const ws = new WebSocket("ws://localhost:3000");

ws.onopen = () => {
  console.log("WebSocket connection opened");
};

ws.onerror = (err) => {
  console.error("WebSocket error:", err);
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
        canvas.toBlob((blob) => {
          if (ws.readyState === WebSocket.OPEN) {
            // Send Blob directly (browser will handle binary)
            ws.send(blob);
            console.log("Frame sent, size:", blob.size);
          }
        }, "image/jpeg", 0.7);

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
