"""
Samanta - Agent vocal pour réservations d'hôtels de luxe
Utilise OpenAI Realtime API (WebSocket) pour une interaction vocale fluide.
"""

import os
import json
import base64
import asyncio
import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()

# Configuration
SAMPLE_RATE = 24000  # OpenAI Realtime API native rate
CHANNELS = 1
DATABASE_PATH = "database_hotel.txt"
MODEL = "gpt-4o-realtime-preview-2024-10-01"

def load_database() -> str:
    """Charge la base de données des hôtels."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, DATABASE_PATH)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Base de données non trouvée."

class AudioHandler:
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        self.stream = None

    def start_stream(self):
        """Démarre les flux audio (entrée et sortie)."""
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            callback=self.callback,
            blocksize=2048 # Latency tuning
        )
        self.stream.start()

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        
        # Input: Envoyer l'audio capturé vers la queue d'entrée
        # On utilise call_soon_threadsafe pour interagir avec l'event loop asyncio depuis le thread audio
        self.loop.call_soon_threadsafe(self.input_queue.put_nowait, indata.copy().tobytes())

        # Output: Lire l'audio depuis la queue de sortie et le jouer
        try:
            # Note: getting from asyncio queue in a sync callback is tricky. 
            # We use a thread-safe approach or a simpler buffer. 
            # For simplicity in this structure, we might need a sync Queue for output or careful handling.
            # Let's use a simpler approach: Pre-buffer in the main loop or use a threading.Queue for output.
            pass 
        except Exception:
            pass
        
        # Pour l'output, une approche robuste avec sounddevice et asyncio est complexe.
        # On va simplifier: on utilise des flux séparés ou un buffer partagé.
        # Ici, pour faire simple, on va laisser le output vide (silence) par défaut
        # et on implémentera la lecture différemment ou via une queue thread-safe.
        outdata.fill(0)

# Re-thinking AudioHandler for simplicity and correctness with asyncio/threads
import queue

class AudioStream:
    def __init__(self):
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = queue.Queue()
        self.stream = None

    def callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)
        
        # Input processing
        loop = asyncio.get_event_loop_policy().get_event_loop()
        # Warning: Getting the loop here might be unsafe if called from another thread.
        # Better to pass the loop in __init__ or use call_soon_threadsafe on a stored loop.
        
        # Output processing
        try:
            data = self.audio_output_queue.get_nowait()
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
            else:
                outdata[:] = data[:len(outdata)]
                # Handle leftover? For simplicity, we assume blocks match or we pad.
                # Ideally we need a ring buffer.
        except queue.Empty:
            outdata.fill(0)
            
        # Input
        # We need to bridge to asyncio. creating a task or call_soon_threadsafe
        # But we need access to the loop. 
        pass

# Let's use a simpler pattern: Main loop handles websocket, separate threads handle audio IO if needed, 
# or use sounddevice's blocking read/write in run_in_executor? 
# No, callbacks are better for low latency.

class RealtimeAgent:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.url = f"wss://api.openai.com/v1/realtime?model={MODEL}"
        self.database = load_database()
        
        # Audio queues
        self.input_audio_queue = asyncio.Queue()
        self.output_audio_queue = queue.Queue() # Thread-safe for SD callback
        self.output_buffer = bytearray() # Buffer pour l'audio sortant
        
        self.loop = None

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Audio error: {status}")
            
        # Input: Push to asyncio queue via threadsafe call
        if self.loop:
            # Conversion explicite pour être sûr (si dtype='int16', c'est déjà bon, mais on vérifie)
            audio_bytes = indata.tobytes()
            self.loop.call_soon_threadsafe(self.input_audio_queue.put_nowait, audio_bytes)
            
            # Debug volume
            # Calculer l'amplitude moyenne pour voir si le micro capte quelque chose
            volume = np.abs(indata).mean()
            if volume > 500: # Seuil arbitraire pour int16
                 print(f"🎤 Vol: {int(volume)}", end="\r", flush=True)
            
        # Output: Pull from thread-safe queue and manage buffer
        # volume = np.linalg.norm(indata) / len(indata)
        # if volume < 50: print("Low volume", end="\r") # int16 range is +/- 32767
        
        # Output: Pull from thread-safe queue and manage buffer
        bytes_needed = frames * 2 # int16 = 2 bytes
        
        # Remplir le buffer avec les données de la queue
        try:
            while len(self.output_buffer) < bytes_needed:
                chunk = self.output_audio_queue.get_nowait()
                self.output_buffer.extend(chunk)
        except queue.Empty:
            pass
            
        # Extraire les données nécessaires
        if len(self.output_buffer) >= bytes_needed:
            data = self.output_buffer[:bytes_needed]
            del self.output_buffer[:bytes_needed]
            outdata[:] = np.frombuffer(data, dtype=np.int16).reshape((frames, CHANNELS))
        else:
            # Pas assez de données, on remplit ce qu'on a et on pad avec du silence
            valid_len = len(self.output_buffer)
            # Remplir avec le buffer disponible
            if valid_len > 0:
                outdata[:valid_len//2] = np.frombuffer(self.output_buffer, dtype=np.int16).reshape((-1, CHANNELS))
                self.output_buffer.clear()
            # Padding silence
            outdata[valid_len//2:] = 0

    async def run(self):
        self.loop = asyncio.get_running_loop()
        
        print(f"Connexion à {self.url}...")
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "OpenAI-Beta": "realtime=v1"
        }
        
        async with websockets.connect(self.url, additional_headers=headers) as ws:
            print("✅ Connecté à OpenAI Realtime API!")
            
            # Initialisation de la session
            await self.initialize_session(ws)
            
            # Démarrage audio
            stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='int16',
                callback=self.audio_callback,
                blocksize=2048
            )
            stream.start()
            
            print("\n🗣️  Parlez maintenant (Ctrl+C pour arrêter)...")
            
            # Tasks
            receive_task = asyncio.create_task(self.receive_loop(ws))
            send_task = asyncio.create_task(self.send_loop(ws))
            
            try:
                await asyncio.gather(receive_task, send_task)
            except asyncio.CancelledError:
                pass
            finally:
                stream.stop()
                stream.close()

    async def initialize_session(self, ws):
        """Configure la session avec le prompt système."""
        system_prompt = f"""Tu es Samanta, une assistante vocale chaleureuse et professionnelle spécialisée dans les réservations d'hôtels de luxe.
Ta base de données:
{self.database}
Réponds de manière concise, conversationnelle et naturelle.
"""
        event = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": system_prompt,
                "voice": "shimmer",  # Voix féminine
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }
        await ws.send(json.dumps(event))

    async def send_loop(self, ws):
        """Envoie l'audio du microphone au websocket."""
        try:
            while True:
                audio_bytes = await self.input_audio_queue.get()
                base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_audio
                }
                await ws.send(json.dumps(event))
        except Exception as e:
            print(f"Erreur d'envoi: {e}")

    async def receive_loop(self, ws):
        """Reçoit les événements du websocket (audio, texte)."""
        try:
            async for message in ws:
                event = json.loads(message)
                event_type = event.get("type")
                
                # Debug events
                if event_type not in ["response.audio.delta", "input_audio_buffer.append"]:
                     print(f"Event: {event_type}")

                if event_type == "session.created":
                    print("Session créée.")
                
                elif event_type == "session.updated":
                    print("Session configurée.")

                elif event_type == "input_audio_buffer.speech_started":
                    print("\n[VAD] Parole détectée...", end="", flush=True)
                
                elif event_type == "input_audio_buffer.speech_stopped":
                    print(" [VAD] Fin de parole.", flush=True)

                elif event_type == "input_audio_buffer.committed":
                    print(" [Audio commited]", flush=True)

                elif event_type == "conversation.item.created":
                    item = event.get("item", {})
                    role = item.get("role")
                    print(f"\n[Item Created] Role: {role}, Type: {item.get('type')}")
                    
                    if role == "user":
                        content = item.get("content", [])
                        if content:
                             print(f" [User content]: {content}")
                    elif role == "assistant":
                        print(" [Réponse en cours...]", end="", flush=True)

                elif event_type == "response.created":
                    print("\n[Response Created]", flush=True)

                elif event_type == "response.audio.delta":
                    audio_b64 = event.get("delta")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        self.output_audio_queue.put(audio_bytes)
                
                elif event_type == "response.audio_transcript.done":
                    print(f"\n🤖 Samanta: {event.get('transcript')}")
                
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    print(f"\n👤 Vous: {event.get('transcript')}")
                
                elif event_type == "error":
                    print(f"\n❌ Erreur API: {event.get('error')}")
                
                elif event_type == "response.done":
                     print(f"\n[Response Done] Status: {event.get('response', {}).get('status')}")
                     output = event.get('response', {}).get('output', [])
                     if output:
                         print(f" [Output items]: {len(output)}")
                     else:
                         print(" [No output content]")
                     
                     # Print usage if available
                     usage = event.get('response', {}).get('usage')
                     if usage:
                         print(f" [Usage]: {usage}")

                elif event_type == "response.output_item.added":
                    item = event.get("item", {})
                    print(f" [Model Item]: {item.get('type')} ({item.get('role')})")

                elif event_type == "response.content_part.added":
                    print(" [Content Part Added]")

        except Exception as e:
            print(f"Erreur de réception: {e}")

if __name__ == "__main__":
    agent = RealtimeAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'agent.")
