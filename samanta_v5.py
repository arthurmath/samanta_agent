"""
Samanta v5 - Agent vocal temps réel destiné aux clients pour réservations d'hôtels de luxe.
Utilise l'API Live de Google Gemini pour une interaction vocale à faible latence.
"""

import asyncio
import csv
import os
import sys
import queue
import threading

# Force UTF-8 encoding for stdout to handle emojis on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
from typing import Callable
import numpy as np
import sounddevice as sd
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()


# Audio configuration
CHUNK_LENGTH_S = 0.05  # 50ms chunks
SAMPLE_RATE_INPUT = 16000  # Input sample rate (Gemini prefers 16kHz)
SAMPLE_RATE_OUTPUT = 24000  # Output sample rate (Gemini outputs 24kHz)
FORMAT = np.int16
CHANNELS = 1
PREBUFFER_CHUNKS = 3  # Number of chunks to buffer before playback
FADE_OUT_MS = 12  # Fade out duration in ms
ENERGY_THRESHOLD = 0.02  # Seuil RMS pour détection de parole (ajustable)
PLAYBACK_ECHO_MARGIN = 0.01  # Marge pour filtrer l'écho pendant la lecture

# Gemini model configuration
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
VOICE_NAME = "Aoede"  # Options: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr


data_path = "database/french"
data_path = "database/english"


def load_database(path: str, base: str = data_path) -> str:
    """Charge la base de données d'un hôtel."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, base, path)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


# Définition des outils pour l'agent (format Gemini)
def rechercher_hotel(nom_hotel: str) -> str:
    """Recherche les informations sur un hôtel spécifique."""
    nom_lower = nom_hotel.lower()
    if "royal" in nom_lower or "palace" in nom_lower or "paris" in nom_lower:
        return load_database("hotels/paris.txt")
    elif "azur" in nom_lower or "nice" in nom_lower or "méditerranée" in nom_lower or "mediterranee" in nom_lower:
        return load_database("hotels/nice.txt")
    elif "mont" in nom_lower or "blanc" in nom_lower or "chamonix" in nom_lower:
        return load_database("hotels/chamonix.txt")
    return "Hôtel non trouvé."


def reserver_hotel(nom_hotel: str, nom_client: str, date_arrivee: str, date_depart: str, nombre_personnes: int) -> str:
    """Réserve un hôtel."""
    csv_file = f"{data_path}/reservations.csv"
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["nom_hotel", "nom_client", "date_arrivee", "date_depart", "nombre_personnes"])
        writer.writerow([nom_hotel, nom_client, date_arrivee, date_depart, nombre_personnes])
    
    return f"Hôtel {nom_hotel} réservé du {date_arrivee} au {date_depart} pour {nombre_personnes} personnes."


def annuler_reservation(nom_hotel: str, nom_client: str, date_arrivee: str, date_depart: str, nombre_personnes: int) -> str:
    """Annule une réservation."""
    csv_file = f"{data_path}/reservations.csv"
    if not os.path.exists(csv_file):
        return "Aucune réservation trouvée."
    
    rows = []
    reservation_found = False
    with open(csv_file, "r", newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            rows.append(header)
        
        for row in reader:
            if (row[0] == nom_hotel and row[1] == nom_client and 
                row[2] == date_arrivee and row[3] == date_depart and 
                int(row[4]) == nombre_personnes):
                reservation_found = True
            else:
                rows.append(row)
    
    if not reservation_found:
        return "Réservation non trouvée dans la base de données."
    
    with open(csv_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    return f"Réservation de l'hôtel {nom_hotel} pour {nom_client} du {date_arrivee} au {date_depart} annulée avec succès."


def obtenir_informations_reservation(nom_client: str) -> str:
    """Obtient les informations d'une réservation."""
    csv_file = f"{data_path}/reservations.csv"
    if not os.path.exists(csv_file):
        return "Aucune réservation trouvée."
    with open(csv_file, "r", newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row[1] == nom_client:
                return f"Réservation de l'hôtel {row[0]} pour {row[1]} du {row[2]} au {row[3]} pour {row[4]} personnes."
    return "Aucune réservation trouvée pour ce client."


def terminer_conversation() -> str:
    """Termine la conversation."""
    return "Conversation terminée"


# Tool function mapping
TOOL_FUNCTIONS: dict[str, Callable] = {
    "search_hotel": rechercher_hotel,
    "book_hotel": reserver_hotel,
    "cancel_booking": annuler_reservation,
    "get_booking_information": obtenir_informations_reservation,
    "end_conversation": terminer_conversation,
}


# Tool declarations for Gemini
TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_hotel",
                description="Recherche les informations sur un hôtel spécifique. Prend en entrée le nom de l'hôtel.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "nom_hotel": types.Schema(
                            type=types.Type.STRING,
                            description="Le nom de l'hôtel à rechercher"
                        )
                    },
                    required=["nom_hotel"]
                )
            ),
            types.FunctionDeclaration(
                name="book_hotel",
                description="Réserve un hôtel. Prend en entrée le nom de l'hôtel, le nom du client, la date d'arrivée, la date de départ et le nombre de personnes.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "nom_hotel": types.Schema(
                            type=types.Type.STRING,
                            description="Le nom de l'hôtel à réserver"
                        ),
                        "nom_client": types.Schema(
                            type=types.Type.STRING,
                            description="Le nom du client"
                        ),
                        "date_arrivee": types.Schema(
                            type=types.Type.STRING,
                            description="La date d'arrivée (format: YYYY-MM-DD)"
                        ),
                        "date_depart": types.Schema(
                            type=types.Type.STRING,
                            description="La date de départ (format: YYYY-MM-DD)"
                        ),
                        "nombre_personnes": types.Schema(
                            type=types.Type.INTEGER,
                            description="Le nombre de personnes"
                        )
                    },
                    required=["nom_hotel", "nom_client", "date_arrivee", "date_depart", "nombre_personnes"]
                )
            ),
            types.FunctionDeclaration(
                name="cancel_booking",
                description="Annule une réservation. Prend en entrée le nom de l'hôtel, le nom du client, la date d'arrivée, la date de départ et le nombre de personnes.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "nom_hotel": types.Schema(
                            type=types.Type.STRING,
                            description="Le nom de l'hôtel"
                        ),
                        "nom_client": types.Schema(
                            type=types.Type.STRING,
                            description="Le nom du client"
                        ),
                        "date_arrivee": types.Schema(
                            type=types.Type.STRING,
                            description="La date d'arrivée (format: YYYY-MM-DD)"
                        ),
                        "date_depart": types.Schema(
                            type=types.Type.STRING,
                            description="La date de départ (format: YYYY-MM-DD)"
                        ),
                        "nombre_personnes": types.Schema(
                            type=types.Type.INTEGER,
                            description="Le nombre de personnes"
                        )
                    },
                    required=["nom_hotel", "nom_client", "date_arrivee", "date_depart", "nombre_personnes"]
                )
            ),
            types.FunctionDeclaration(
                name="get_booking_information",
                description="Obtient les informations d'une réservation. Prend en entrée le nom du client.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "nom_client": types.Schema(
                            type=types.Type.STRING,
                            description="Le nom du client"
                        )
                    },
                    required=["nom_client"]
                )
            ),
            types.FunctionDeclaration(
                name="end_conversation",
                description="Termine la conversation avec le client.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={},
                    required=[]
                )
            )
        ]
    )
]


class SamantaGeminiAgent:
    """Agent vocal temps réel pour les réservations d'hôtels avec Gemini."""
    
    def __init__(self) -> None:
        self.session = None
        self.audio_stream: sd.InputStream | None = None
        self.audio_player: sd.OutputStream | None = None
        self.recording = False
        self.should_exit = False
        
        # Audio input/output queues
        self.input_queue: queue.Queue[bytes] = queue.Queue()
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.current_audio_chunk: np.ndarray | None = None
        self.chunk_position = 0
        self.prebuffering = True
        
        # Interruption handling and RMS tracking
        self.interrupt_event = threading.Event()
        self.fading = False
        self.fade_samples = int(SAMPLE_RATE_OUTPUT * (FADE_OUT_MS / 1000.0))
        self.fade_done_samples = 0
        self.fade_total_samples = 0
        self.playback_rms = 0.0
        
        # Protection contre auto-interruption au démarrage
        self.warmup_complete = False
        self.warmup_chunks_needed = 120  # ~6 seconds of warmup (120 chunks * 50ms)
        
        # Client Gemini
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.system_prompt = load_database("syst_prompt.txt")
        self.system_prompt += "\n\n" + load_database("infos_generales.txt")
        self.system_prompt += "\n\nIMPORTANT INSTRUCTIONS:\n"
        self.system_prompt += "1. You are a voice assistant. Your responses will be read out loud.\n"
        self.system_prompt += "2. DO NOT output any internal reasoning, thoughts, or 'thinking' process.\n"
        self.system_prompt += "3. DO NOT use markdown formatting like **bold** or *italics*.\n"
        self.system_prompt += "4. Respond directly to the user with spoken text only.\n"
        self.system_prompt += "5. Keep responses concise and conversational.\n"
        self.system_prompt += "6. START by greeting the user with: 'Hello and welcome! I'm Samanta, your personal assistant for SBM hotel bookings. How can I help you today?'"

    def _compute_rms(self, samples: np.ndarray) -> float:
        """Calcule l'énergie RMS des échantillons."""
        if samples.size == 0:
            return 0.0
        x = samples.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x)))

    def _update_playback_rms(self, samples: np.ndarray) -> None:
        """Met à jour l'estimation d'énergie de lecture pour le filtrage d'écho."""
        sample_rms = self._compute_rms(samples)
        self.playback_rms = 0.9 * self.playback_rms + 0.1 * sample_rms
        
        # Marquer le warmup comme terminé après avoir joué suffisamment de chunks
        if not self.warmup_complete and self.warmup_chunks_needed > 0:
            self.warmup_chunks_needed -= 1
            if self.warmup_chunks_needed == 0:
                self.warmup_complete = True
                print("🔓 Interruption detection activated")

    def _input_callback(self, indata, frames, time_info, status) -> None:
        """Callback pour la capture audio."""
        if status:
            print(f"Status entrée audio: {status}")
        self.input_queue.put(indata.copy().tobytes())

    def _output_callback(self, outdata, frames: int, time_info, status) -> None:
        """Callback pour la sortie audio avec gestion des interruptions."""
        if status:
            print(f"Status sortie audio: {status}")

        # Handle interruption with fade-out
        if self.interrupt_event.is_set():
            outdata.fill(0)
            if self.current_audio_chunk is None:
                # Clear the queue
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                self.prebuffering = True
                self.interrupt_event.clear()
                return

            if not self.fading:
                self.fading = True
                self.fade_done_samples = 0
                remaining_in_chunk = len(self.current_audio_chunk) - self.chunk_position
                self.fade_total_samples = min(self.fade_samples, max(0, remaining_in_chunk))

            samples = self.current_audio_chunk
            samples_filled = 0
            while samples_filled < frames and self.fade_done_samples < self.fade_total_samples:
                remaining_output = frames - samples_filled
                remaining_fade = self.fade_total_samples - self.fade_done_samples
                n = min(remaining_output, remaining_fade)
                src = samples[self.chunk_position:self.chunk_position + n].astype(np.float32)
                idx = np.arange(self.fade_done_samples, self.fade_done_samples + n, dtype=np.float32)
                gain = 1.0 - (idx / float(self.fade_total_samples))
                ramped = np.clip(src * gain, -32768.0, 32767.0).astype(np.int16)
                outdata[samples_filled:samples_filled + n, 0] = ramped
                self._update_playback_rms(ramped)
                samples_filled += n
                self.chunk_position += n
                self.fade_done_samples += n

            if self.fade_done_samples >= self.fade_total_samples:
                self.current_audio_chunk = None
                self.chunk_position = 0
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                self.fading = False
                self.prebuffering = True
                self.interrupt_event.clear()
            return

        # Normal audio buffer filling
        outdata.fill(0)
        samples_filled = 0
        while samples_filled < frames:
            if self.current_audio_chunk is None:
                try:
                    # Prebuffering: wait for enough chunks
                    if self.prebuffering and self.output_queue.qsize() < PREBUFFER_CHUNKS:
                        break
                    self.prebuffering = False
                    self.current_audio_chunk = self.output_queue.get_nowait()
                    self.chunk_position = 0
                except queue.Empty:
                    break

            remaining_output = frames - samples_filled
            samples = self.current_audio_chunk
            remaining_chunk = len(samples) - self.chunk_position
            samples_to_copy = min(remaining_output, remaining_chunk)

            if samples_to_copy > 0:
                chunk_data = samples[self.chunk_position:self.chunk_position + samples_to_copy]
                outdata[samples_filled:samples_filled + samples_to_copy, 0] = chunk_data
                self._update_playback_rms(chunk_data)
                samples_filled += samples_to_copy
                self.chunk_position += samples_to_copy

            if self.chunk_position >= len(samples):
                self.current_audio_chunk = None
                self.chunk_position = 0

    async def run(self) -> None:
        """Lance l'agent Samanta avec Gemini."""
        print("=" * 60)
        print("🏨 Samanta v5 - Agent Vocal SBM (Gemini)")
        print("=" * 60)
        print("\nCommande Ctrl+C pour quitter")

        # Configure audio output
        chunk_size_output = int(SAMPLE_RATE_OUTPUT * CHUNK_LENGTH_S)
        self.audio_player = sd.OutputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE_OUTPUT,
            dtype=FORMAT,
            callback=self._output_callback,
            blocksize=chunk_size_output,
        )
        self.audio_player.start()

        # Configure audio input
        chunk_size_input = int(SAMPLE_RATE_INPUT * CHUNK_LENGTH_S)
        self.audio_stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE_INPUT,
            dtype=FORMAT,
            callback=self._input_callback,
            blocksize=chunk_size_input,
        )
        self.audio_stream.start()
        self.recording = True

        retry_count = 0
        max_retries = 5

        while not self.should_exit and retry_count < max_retries:
            try:
                print("Connexion en cours...")
                # Configure the Live API session
                config = types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    system_instruction=types.Content(
                        parts=[types.Part(text=self.system_prompt)]
                    ),
                    tools=TOOLS,
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE_NAME
                            )
                        )
                    ),
                )
                
                async with self.client.aio.live.connect(model=MODEL, config=config) as session:
                    self.session = session
                    print("✅ Connecté à Gemini!")
                    retry_count = 0 # Reset retry count on successful connection
                    
                    # Start audio capture task FIRST (keeps session alive)
                    audio_task = asyncio.create_task(self.capture_audio())
                    
                    # Start event processing task
                    event_task = asyncio.create_task(self.process_events())
                    
                    # Small delay to ensure audio streaming has started
                    await asyncio.sleep(0.1)
                    
                    # Démarrer la conversation avec un message d'accueil
                    phrase_accueil = load_database("sentence.txt")
                    # Clean the phrase to be just the text
                    if ":" in phrase_accueil:
                        phrase_accueil = phrase_accueil.split(":", 1)[1].strip().strip('"')

                    await session.send_client_content(
                        turns=[types.Content(
                            role="user",
                            parts=[types.Part(text=phrase_accueil)]
                        )]
                    )
                    
                    print("\n🎤 Début de la conversation")
                    
                    # Wait for tasks
                    await asyncio.gather(audio_task, event_task)

            except Exception as e:
                if self.should_exit:
                    break
                retry_count += 1
                print(f"⚠️ Erreur session (essai {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(2)
                else:
                    print("❌ Nombre maximal de tentatives atteint.")
            finally:
                self.session = None

        time.sleep(2)
        if self.audio_player and self.audio_player.active:
            self.audio_player.stop()
        if self.audio_player:
            self.audio_player.close()
        if self.audio_stream and self.audio_stream.active:
            self.audio_stream.stop()
        if self.audio_stream:
            self.audio_stream.close()
        print("\n👋 Bye-bye!")

    async def capture_audio(self) -> None:
        """Capture l'audio du microphone et l'envoie à Gemini avec filtrage d'écho."""
        try:
            while self.recording and self.session and not self.should_exit:
                try:
                    audio_bytes = self.input_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                # Check if assistant is currently playing audio
                assistant_playing = self.current_audio_chunk is not None or not self.output_queue.empty()
                
                # MIME type must include sample rate for PCM audio
                audio_mime_type = f"audio/pcm;rate={SAMPLE_RATE_INPUT}"
                
                if assistant_playing:
                    # During playback: filter to prevent echo/self-interruption
                    samples = np.frombuffer(audio_bytes, dtype=np.int16)
                    mic_rms = self._compute_rms(samples)
                    # Dynamic threshold based on playback level
                    threshold = max(ENERGY_THRESHOLD, self.playback_rms * 1.5 + PLAYBACK_ECHO_MARGIN)
                    
                    # During warmup, always send audio but don't trigger interruption
                    if not self.warmup_complete:
                        await self.session.send_realtime_input(
                            audio={
                                "data": audio_bytes,
                                "mime_type": audio_mime_type
                            }
                        )
                    elif mic_rms >= threshold:
                        # User is speaking louder than playback - send audio
                        await self.session.send_realtime_input(
                            audio={
                                "data": audio_bytes,
                                "mime_type": audio_mime_type
                            }
                        )
                    else:
                        # Still send audio to keep session alive, but below threshold
                        await self.session.send_realtime_input(
                            audio={
                                "data": audio_bytes,
                                "mime_type": audio_mime_type
                            }
                        )
                else:
                    # Not playing: always send audio to keep session alive
                    await self.session.send_realtime_input(
                        audio={
                            "data": audio_bytes,
                            "mime_type": audio_mime_type
                        }
                    )
                
                await asyncio.sleep(0)
        
        except asyncio.CancelledError:
            pass  # Normal cancellation

        except Exception as e:
            if not self.should_exit:
                print(f"Erreur capture audio: {e}")
        finally:
            pass # L'arrêt des streams est géré dans run()

    async def process_events(self) -> None:
        """Process events from the Gemini session."""
        try:
            # Keep receiving turns in a loop
            while not self.should_exit:
                try:
                    # Get the next turn from the session
                    turn = self.session.receive()
                    async for response in turn:
                        await self._handle_response(response)
                        
                        if self.should_exit:
                            self.recording = False
                            break
                    
                    # After turn completes, clear the output queue (handle interruptions)
                    if not self.should_exit:
                        # Small delay before processing next turn
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    if self.should_exit:
                        break
                    print(f"⚠️ Erreur réception: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(0.5)
            
            print("ℹ️  Session terminée")
                    
        except asyncio.CancelledError:
            print("ℹ️  Arrêt demandé")
        except Exception as e:
            print(f"❌ Erreur événement critique: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.recording = False
            self.should_exit = True

    async def _handle_response(self, response) -> None:
        """Handle a response from Gemini."""
        try:
            
            # Check for server content
            if response.server_content:
                server_content = response.server_content
                
                # Handle model turn (audio/text response)
                if server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        # Handle audio output
                        if part.inline_data and part.inline_data.mime_type and part.inline_data.mime_type.startswith("audio/"):
                            audio_data = part.inline_data.data
                            # Convert bytes to numpy array and queue it
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            self.output_queue.put(audio_array)
                        
                        # Handle text (if any) - suppress "thinking" parts
                        if part.text:
                            text = part.text.strip()
                            # Filter out internal reasoning (usually starts with ** or is very short/meta)
                            if text and not (text.startswith("**") or text.startswith("User:") or text.startswith("Model:")):
                                print(f"💬 Samanta: {text}")
                
                # Handle turn complete
                if server_content.turn_complete:
                    pass  # Turn finished
                
                # Handle interrupted
                if server_content.interrupted:
                    # Ignore interruptions during warmup to prevent false positives
                    if not self.warmup_complete:
                        # During warmup, just reset prebuffering but don't trigger interrupt
                        self.prebuffering = True
                    else:
                        print("🎤 Interruption détectée")
                        # Trigger interruption handling
                        self.interrupt_event.set()
                        # Do not set should_exit, just clear queue and continue listening
            
            # Handle tool calls
            if response.tool_call:
                await self._handle_tool_call(response.tool_call)
                
        except Exception as e:
            print(f"Erreur traitement réponse: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_tool_call(self, tool_call) -> None:
        """Handle a tool call from Gemini."""
        try:
            for function_call in tool_call.function_calls:
                tool_name = function_call.name
                tool_args = dict(function_call.args) if function_call.args else {}
                
                print(f"🔧 Outil utilisé: {tool_name}")
                
                # Execute the tool
                if tool_name in TOOL_FUNCTIONS:
                    try:
                        result = TOOL_FUNCTIONS[tool_name](**tool_args)
                        print(f"✅ Action effectuée")
                        
                        # Check if it's the end conversation tool
                        if tool_name == "end_conversation":
                            self.should_exit = True
                        
                        # Send the tool response back to Gemini
                        await self.session.send_tool_response(
                            function_responses=[
                                types.FunctionResponse(
                                    name=tool_name,
                                    id=function_call.id,
                                    response={"result": result}
                                )
                            ]
                        )
                    except Exception as e:
                        print(f"❌ Erreur outil {tool_name}: {e}")
                        await self.session.send_tool_response(
                            function_responses=[
                                types.FunctionResponse(
                                    name=tool_name,
                                    id=function_call.id,
                                    response={"error": str(e)}
                                )
                            ]
                        )
                else:
                    print(f"❌ Outil inconnu: {tool_name}")
                    
        except Exception as e:
            print(f"Erreur tool call: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    samanta = SamantaGeminiAgent()
    try:
        asyncio.run(samanta.run())
    except KeyboardInterrupt:
        sys.exit(0)
