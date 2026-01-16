"""
Samanta v6 - Agent vocal temps réel pour le marketing/consulting.
Utilise l'API Live de Google Gemini pour une interaction vocale à faible latence.
"""

import asyncio
import os
import sys
import queue
import threading

# Force UTF-8 encoding for stdout to handle emojis on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

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
ENERGY_THRESHOLD = 100  # Seuil RMS pour détection de parole
PLAYBACK_ECHO_MARGIN = 0.01  # Marge pour filtrer l'écho pendant la lecture

# Gemini model configuration
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
VOICE_NAME = "Aoede"  # Options: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr


SYSTEM_PROMPT = """You are Samanta, a friendly and professional Luxury Consultant specialized in everything SBM, an expert strategist in luxury hostels and restaurants, 
aware of trends with a creative and marketing mind. Aware of numbers, what works and what doesn't, with a strong background in brand strategy in advertising 
and very connected to innovation in the digital spaces, including AI. SBM (Societe des Bains de Mer) is a large company based in Monaco that owns several luxury hotels and restaurants in France and Monaco (Monte Carlo).
 
Your role:
- Advise, inform and empower teams with strategic planning and knowledge in luxury;
- Give marketing insights in e-commerce best practices and maximise conversion;
- Describe shortly your role as a Expert Strategic AI Consultant in the world of Luxury;
- You are well informed in the world of SEO, GEO and what it takes to make websites and e-commerce platforms cover;
- You are there to support the Team at Niji succeed in creating a new digital experience for SBM with the best information available in the market. Always double check source is correct
- Be concise but informative (your answers will be read aloud).
- Always respond in British English with an elegant and warm tone.
- Your boss is Chris De Abreu, International Executive Creative Director at Niji (French digital consulting firm), he is the one mentoring you.
- Don't start talking unless we invite you to do so.

IMPORTANT INSTRUCTIONS:
1. You are a voice assistant. Your responses will be read out loud.
2. DO NOT output any internal reasoning, thoughts, or 'thinking' process.
3. DO NOT use markdown formatting like **bold** or *italics*.
4. Respond directly to the user with spoken text only.
5. Keep responses concise and conversational.
"""


class SamantaGeminiAgent:
    """Agent vocal temps réel pour le marketing/consulting avec Gemini."""
    
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
        self.warmup_chunks_needed = 120  # ~3 seconds of warmup (60 chunks * 50ms)
        
        # Track if agent is currently speaking
        self.agent_speaking = False
        
        # Client Gemini
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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
        print("🏨 Samanta v6 - Agent Vocal Marketing (Gemini)")
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
                        parts=[types.Part(text=SYSTEM_PROMPT)]
                    ),
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
                    print("🎤 Listening... (speak to Samanta)")
                    retry_count = 0  # Reset retry count on successful connection
                    
                    # Start audio capture task
                    audio_task = asyncio.create_task(self.capture_audio())
                    
                    # Start event processing task
                    event_task = asyncio.create_task(self.process_events())
                    
                    # Wait for tasks
                    await asyncio.gather(audio_task, event_task)

            except Exception as e:
                print(f"Exception: {e}")
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

                try:
                    # Check if assistant is currently playing audio
                    assistant_playing = self.current_audio_chunk is not None or not self.output_queue.empty()
                    
                    # MIME type must include sample rate for PCM audio
                    audio_mime_type = f"audio/pcm;rate={SAMPLE_RATE_INPUT}"
                    
                    if assistant_playing:
                        # During playback: check if user is interrupting
                        samples = np.frombuffer(audio_bytes, dtype=np.int16)
                        mic_rms = self._compute_rms(samples)
                        # Seuil plus strict: doit être significativement plus fort que l'écho
                        playback_gate = max(ENERGY_THRESHOLD, self.playback_rms * 2.0 + PLAYBACK_ECHO_MARGIN)
                        
                        # During warmup, send audio but don't trigger interruption
                        if not self.warmup_complete:
                            await self.session.send_realtime_input(
                                audio={
                                    "data": audio_bytes,
                                    "mime_type": audio_mime_type
                                }
                            )
                        elif mic_rms >= playback_gate:
                            # User is speaking louder than playback - send audio and trigger interrupt
                            self.interrupt_event.set()
                            await self.session.send_realtime_input(
                                audio={
                                    "data": audio_bytes,
                                    "mime_type": audio_mime_type
                                }
                            )
                        else:
                            # Still send audio to keep session alive, but don't interrupt
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
                except Exception as send_error:
                    if not self.should_exit:
                        print(f"⚠️ Erreur envoi audio: {send_error}")
                        # If we can't send audio, the session might be dead
                        if "session" in str(send_error).lower() or "closed" in str(send_error).lower():
                            break
                
                await asyncio.sleep(0)
        
        except asyncio.CancelledError:
            pass  # Normal cancellation

        except Exception as e:
            if not self.should_exit:
                print(f"❌ Erreur capture audio: {e}")
                import traceback
                traceback.print_exc()
        finally:
            pass  # L'arrêt des streams est géré dans run()

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
                            if not self.agent_speaking:
                                print("💬 Samanta is speaking...")
                                self.agent_speaking = True
                            audio_data = part.inline_data.data
                            # Convert bytes to numpy array and queue it
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                            self.output_queue.put(audio_array)
                
                # Handle turn complete
                if server_content.turn_complete:
                    if self.agent_speaking:
                        self.agent_speaking = False
                        print("🎤 Listening... (speak to Samanta)")
                
                # Handle interrupted
                if server_content.interrupted:
                    # Ignore interruptions during warmup to prevent false positives
                    if not self.warmup_complete:
                        # During warmup, just reset prebuffering but don't trigger interrupt
                        self.prebuffering = True
                    else:
                        if self.agent_speaking:
                            print("🎤 Interruption - Voice detected")
                            self.agent_speaking = False
                        # Trigger interruption handling
                        self.interrupt_event.set()
                        self.prebuffering = True
                    
        except Exception as e:
            print(f"Error processing response: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    samanta = SamantaGeminiAgent()
    try:
        asyncio.run(samanta.run())
    except KeyboardInterrupt:
        sys.exit(0)
