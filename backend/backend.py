import asyncio
import websockets
import datetime
import os
import random
import json




SAVE_DIR = "frames"
os.makedirs(SAVE_DIR, exist_ok=True)

INSTRUMENT = ["Piano","Kickdrum", "Snare", "High Hat"]##,"Kickdrum", "Snare", "High Hat"
BPM = ["100", "150", "200"]
NOTE = ["A", "B", "C", "D","E","F","G"]

async def handle_connection(websocket):
    print("Client connected")
    
    async def send_data():
        try:
            while True:
                instrument_type = random.choice(INSTRUMENT)
                BPM_type = random.choice(BPM)
                note_type = random.choice(NOTE)
                data = {"instrument": "High Hat", "bpm": BPM_type, "note":note_type}
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
    await asyncio.gather(send_data()) ##, receive_frames()

async def main():
    server = await websockets.serve(handle_connection, "localhost", 3000)
    print("WebSocket server started on ws://localhost:3000")
    await server.wait_closed()

asyncio.run(main())
