import os
import numpy as np
import sounddevice as sd
import time
import asyncio
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer

from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    
voices = [
    "alloy", # 6
    "marin", # 7
    "sage", # 5
    "nova", # 7 moins rapide
    # "fable", # 4
    # "shimmer", # 4
    # "echo", # 3
    # "ash", # 1
    # "ballad", # 1
    # "coral", # 2
    # "onyx", # 0
    # "verse", # 1
    # "cedar" # 1
]

async def text_to_speech(voice: str):
    async with aclient.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input="Bonjour et bienvenue! Je suis Samanta, votre assistante personnelle pour les réservations d'hôtels SBM. Comment puis-je vous aider aujourd'hui?",
        instructions="Speak in a professionnal and warm tone.",
        response_format="pcm",
        speed=1.1
    ) as response:
        await LocalAudioPlayer().play(response)



for voice in voices:
    print(voice)
    asyncio.run(text_to_speech(voice))
    







# # Save audio as mp3 file

# from pathlib import Path
# speech_file_path = Path(__file__).parent / "speech.mp3"

# with client.audio.speech.with_streaming_response.create(
#     model="gpt-4o-mini-tts",
#     voice="coral",
#     input="Today is a wonderful day to build something people love!",
#     instructions="Speak in a cheerful and positive tone.",
# ) as response:
#     response.stream_to_file(speech_file_path)







# # Faster sync or async?

# def text_to_speech():
#     response = client.audio.speech.create(
#         model="gpt-4o-mini-tts",
#         voice="nova",  # Voix féminine naturelle
#         input="Today is a wonderful day to build something people love!",
#         response_format="pcm",
#         speed=1.1  # Légèrement plus rapide pour plus de naturel
#     )
    
#     # Lecture directe du PCM
#     audio_data = np.frombuffer(response.content, dtype=np.int16)
#     audio_float = audio_data.astype(np.float32) / 32768.0
    
#     sd.play(audio_float, samplerate=24000)
#     sd.wait()



# async def atext_to_speech():
#     async with aclient.audio.speech.with_streaming_response.create(
#         model="gpt-4o-mini-tts",
#         voice="nova",
#         input="Today is a wonderful day to build something people love!",
#         instructions="Speak in a professionnal and warm tone.",
#         response_format="pcm",
#         speed=1.1
#     ) as response:
#         await LocalAudioPlayer().play(response)



# if __name__ == "__main__":
#     start = time.time()
#     text_to_speech()
#     end = time.time()
#     print(f"Sync: {end - start:.2f} seconds")

#     start = time.time()
#     asyncio.run(atext_to_speech())
#     end = time.time()
#     print(f"Async: {end - start:.2f} seconds")
