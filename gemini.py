import asyncio
from google import genai
import pyaudio
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# --- pyaudio config ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# --- Voice Activity Detection config ---
ENERGY_THRESHOLD = 0.02  # RMS threshold for voice detection (lower = more sensitive)
PLAYBACK_THRESHOLD_MULTIPLIER = 2.0  # During playback, require 2x the threshold to detect voice

# --- Live API config ---
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful and professional AI assistant.",
}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
pya = pyaudio.PyAudio()
audio_queue_output = asyncio.Queue()
audio_queue_mic = asyncio.Queue(maxsize=5)
audio_stream = None
is_playing = False  # Track if agent is currently speaking

def compute_rms(audio_data):
    """Compute RMS (Root Mean Square) energy of audio data."""
    samples = np.frombuffer(audio_data, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    normalized = samples.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(normalized * normalized)))

async def listen_audio():
    """Listens for audio and puts it into the mic audio queue with voice activity detection."""
    global audio_stream
    mic_info = pya.get_default_input_device_info()
    audio_stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )
    kwargs = {"exception_on_overflow": False} if __debug__ else {}
    while True:
        data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
        
        # Calculate RMS energy of the audio
        rms = compute_rms(data)
        
        # Determine threshold based on whether agent is speaking
        threshold = ENERGY_THRESHOLD
        if is_playing:
            threshold = ENERGY_THRESHOLD * PLAYBACK_THRESHOLD_MULTIPLIER
        
        # Only queue audio if it exceeds the threshold
        # Always send during silence to keep connection alive, but use a minimum send rate
        if rms >= threshold or not is_playing:
            await audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})

async def send_realtime(session):
    """Sends audio from the mic audio queue to the GenAI session."""
    while True:
        msg = await audio_queue_mic.get()
        await session.send_realtime_input(audio=msg)

async def receive_audio(session):
    """Receives responses from GenAI and puts audio data into the speaker audio queue."""
    global is_playing
    while True:
        turn = session.receive()
        async for response in turn:
            if (response.server_content and response.server_content.model_turn):
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        audio_queue_output.put_nowait(part.inline_data.data)
                        is_playing = True

        # Empty the queue on interruption to stop playback
        while not audio_queue_output.empty():
            audio_queue_output.get_nowait()
        is_playing = False

async def play_audio():
    """Plays audio from the speaker audio queue."""
    global is_playing
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
    )
    while True:
        bytestream = await audio_queue_output.get()
        is_playing = True
        await asyncio.to_thread(stream.write, bytestream)
        # Check if queue is empty - if so, we're done playing
        if audio_queue_output.empty():
            is_playing = False

async def run():
    """Main function to run the audio loop."""
    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as live_session:
            print("Connected to Gemini. Start speaking!")
            async with asyncio.TaskGroup() as tg:
                tg.create_task(send_realtime(live_session))
                tg.create_task(listen_audio())
                tg.create_task(receive_audio(live_session))
                tg.create_task(play_audio())
    except asyncio.CancelledError:
        pass
    finally:
        if audio_stream:
            audio_stream.close()
        pya.terminate()
        print("\nConnection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")