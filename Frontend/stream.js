const video = document.getElementById('live-stream');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

// Connect to WebSocket server
const ws = new WebSocket('ws://localhost:5500'); // change port if needed

ws.onopen = () => {
    console.log('WebSocket connection opened');
};

ws.onerror = (err) => {
    console.error('WebSocket error:', err);
};

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;

        // Start sending frames after video metadata is loaded
        video.addEventListener('loadedmetadata', () => {
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
                    // Convert Blob to ArrayBuffer for WebSocket
                    blob.arrayBuffer().then(buffer => {
                        ws.send(buffer);
                    });
                }
            }, 'image/jpeg', 0.7); //0.7 is the quality

            // Schedule next frame
            requestAnimationFrame(sendFrames);
        }

    })
    .catch((error) => {
        console.error("Error accessing camera:", error);
    });
} else {
    console.log("getUserMedia is not supported in this browser.");
}