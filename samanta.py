"""
Samanta - Agent vocal pour réservations d'hôtels de luxe
Utilise OpenAI Whisper (STT), GPT-4 (LLM), et TTS pour une interaction vocale naturelle.
"""

import os
import io
import asyncio
import tempfile
import threading
import queue
import numpy as np
from openai.helpers import LocalAudioPlayer
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from openai import OpenAI, AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
DATABASE_PATH = "database_hotel.txt"

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_database() -> str:
    """Charge la base de données des hôtels."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, DATABASE_PATH)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


async def record_audio(duration: float = None, silence_threshold: float = 0.01, silence_duration: float = 1.2) -> np.ndarray:
    """
    Enregistre l'audio du microphone.
    Si duration est None, arrête automatiquement après silence_duration secondes de silence.
    """
    print("\n🎤 Parlez maintenant... (silence pour terminer)")
    
    audio_data = []
    silence_samples = 0
    samples_for_silence = int(SAMPLE_RATE * silence_duration)
    recording = True
    
    def audio_callback(indata, frames, time, status):
        nonlocal silence_samples, recording
        if status:
            print(f"Status: {status}")
        
        audio_data.append(indata.copy())
        
        # Détection du silence
        volume = np.abs(indata).mean()
        if volume < silence_threshold:
            silence_samples += frames
        else:
            silence_samples = 0
        
        if silence_samples >= samples_for_silence and len(audio_data) > 10:
            recording = False
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, dtype='float32'):
        while recording:
            await asyncio.sleep(0.1)
    
    if audio_data:
        return np.concatenate(audio_data, axis=0)
    return np.array([])


async def speech_to_text(audio_data: np.ndarray) -> str:
    """Convertit l'audio en texte avec Whisper."""
    print("🔄 Transcription en cours...")
    
    # Sauvegarde temporaire en WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        audio_int16 = (audio_data * 32767).astype(np.int16)
        write_wav(tmp_file.name, SAMPLE_RATE, audio_int16)
    
    try:
        with open(tmp_file.name, "rb") as audio_file:
            transcript = await async_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="fr"
            )
        return transcript.text
    finally:
        os.unlink(tmp_file.name)


async def text_to_speech(text: str):
    """Convertit le texte en parole et le lit immédiatement."""

    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=text,
        instructions="Speak in a professionnal and warm tone.",
        response_format="pcm",
        speed=1.1
    ) as response:
        await LocalAudioPlayer().play(response)



async def get_ai_response(user_message: str, database: str, conversation_history: list) -> str:
    """Obtient une réponse de GPT basée sur la question et la base de données."""
    
    system_prompt = f"""Tu es Samanta, une assistante vocale chaleureuse et professionnelle spécialisée dans les réservations d'hôtels de luxe. 
Ton rôle:
- Répondre aux questions des clients sur les hôtels, chambres, restaurants et activités
- Aider à la réservation et fournir des recommandations personnalisées
- Être concise mais informative (tes réponses seront lues à voix haute)
- Toujours répondre en français avec un ton élégant et accueillant
Voici la base de données des hôtels disponibles:
<database>
{database}
</database>
Instructions importantes:
- Réponds de manière naturelle et conversationnelle
- Évite les listes à puces, préfère des phrases fluides
- Si tu ne trouves pas l'information demandée, ne l'invente surtout pas, mais dis poliment que tu ne disposes pas de ces informations.
- Garde tes réponses relativement courtes (2-4 phrases) sauf si plus de détails sont demandés"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    
    response = await async_client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        temperature=1,
        max_completion_tokens=500
    )
    
    return response.choices[0].message.content


async def main():
    """Boucle principale de l'agent Samanta."""
    print("=" * 60)
    print("🏨 Bienvenue! Je suis Samanta, votre assistante pour")
    print("       les réservations d'hôtels SBM.")
    print("=" * 60)
    print("\nDites 'quitter' pour terminer la conversation")
    
    # Charger la base de données
    database = load_database()
    conversation_history = []
    
    # Message de bienvenue
    welcome = "Bonjour et bienvenue! Je suis Samanta, votre assistante personnelle pour les réservations d'hôtels SBM. Comment puis-je vous aider aujourd'hui?"
    print(f"\n💬 Samanta: {welcome}")
    await text_to_speech(welcome)
    
    while True:
        try:
            # input("\n⏎ Appuyez sur Entrée pour parler...")
            
            # Enregistrement
            audio = await record_audio()
            if len(audio) == 0:
                print("❌ Aucun audio détecté.")
                continue
            
            # Transcription
            user_text = await speech_to_text(audio)
            print(f"👤 Vous: {user_text}\n")
            
            if not user_text.strip():
                continue
            
            # Vérifier si l'utilisateur veut quitter
            if any(word in user_text.lower() for word in ["quitter", "quittez", "au revoir", "termine"]):
                goodbye = "Au revoir et merci pour votre visite!"
                print(f"\n💬 Samanta: {goodbye}")
                await text_to_speech(goodbye)
                break
            
            # Obtenir la réponse
            response = await get_ai_response(user_text, database, conversation_history)
            
            # Synthèse vocale
            print(f"💬 Samanta: {response}")
            await text_to_speech(response)

            # Mise à jour de l'historique
            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": response})
            
            # Garder seulement les 20 derniers échanges
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
        except KeyboardInterrupt:
            print("\n\n👋 Interruption détectée. Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())












# def text_to_speech(text: str):
#     """Convertit le texte en parole et le lit immédiatement."""
#     print("🔊 Samanta parle...\n")
    
#     response = client.audio.speech.create(
#         # model="tts-1",
#         model="gpt-4o-mini-tts",
#         voice="nova",  # Voix féminine naturelle
#         input=text,
#         response_format="pcm",
#         speed=1.2  # Légèrement plus rapide pour plus de naturel
#     )
    
#     # Lecture directe du PCM
#     audio_data = np.frombuffer(response.content, dtype=np.int16)
#     audio_float = audio_data.astype(np.float32) / 32768.0
    
#     sd.play(audio_float, samplerate=24000)
#     sd.wait()