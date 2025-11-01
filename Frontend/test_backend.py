import asyncio
import websockets
import datetime
import os
import random
import json




SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)

INSTRUMENT = ["kickdrum", "snare", "high_hat", "piano"]
BPM = ["100", "150", "200"]

async def handle_connection(websocket):
    print("Client connected")
    
    async def send_drum_types():
        try:
            while True:
                instrument_type = random.choice(INSTRUMENT)
                BPM_type = random.choice(BPM)
                data = {"instrument": instrument_type, "bpm": BPM_type}
                await websocket.send(json.dumps(data))  # prefix messages
                await asyncio.sleep(2)
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected from drum sender")

    async def receive_frames():
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
                    with open(filename, "wb") as f:
                        f.write(message)
                    print(f"Saved frame: {filename}")
                else:
                    print("Received non-bytes message")
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected from frame receiver")

    # Run both tasks concurrently
    await asyncio.gather(send_drum_types()) ##, receive_frames()

async def main():
    server = await websockets.serve(handle_connection, "localhost", 3000)
    print("WebSocket server started on ws://localhost:3000")
    await server.wait_closed()

asyncio.run(main())
