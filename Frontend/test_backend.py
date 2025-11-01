import asyncio
import websockets
import datetime
import os

SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)

async def receive_frames(websocket):
    print("Client connected")
    try:
        async for message in websocket:
            if isinstance(message, str):
                print("Warning: received string instead of bytes")
                continue

            elif isinstance(message, bytes):
                print("Received frame, size:", len(message))

            # Save frame to disk
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
            with open(filename, "wb") as f:
                f.write(message)
            print(f"Saved frame: {filename}")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    # Start WebSocket server
    server = await websockets.serve(receive_frames, "localhost", 3000)
    print("WebSocket server started on ws://localhost:3000")
    await server.wait_closed()  # keep server running

# Run the server
asyncio.run(main())
