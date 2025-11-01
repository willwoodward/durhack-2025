// Select the video element
const video = document.getElementById('live-stream');

// Check if the browser supports the MediaDevices API
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Request access to the user's camera
    navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        // Set the video source to the camera stream
        video.srcObject = stream;
    })
    .catch((error) => {
        console.error("Error accessing the camera: ", error);
    });
} else {
    console.log("getUserMedia is not supported in this browser.");
}