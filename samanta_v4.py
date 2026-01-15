"""
Samanta v4 - Agent vocal temps réel pour le marketing/consulting
"""

import asyncio
import csv
import os
import queue
import sys
import threading
from typing import Any
import numpy as np
import sounddevice as sd
import time

from agents import function_tool
from agents.realtime import (
    RealtimeAgent,
    RealtimePlaybackTracker,
    RealtimeRunner,
    RealtimeSession,
    RealtimeSessionEvent,
)
from agents.realtime.model import RealtimeModelConfig

from dotenv import load_dotenv
load_dotenv()


# Audio configuration
CHUNK_LENGTH_S = 0.04  # 40ms
SAMPLE_RATE = 24000
FORMAT = np.int16
CHANNELS = 1
ENERGY_THRESHOLD = 0.1  # Seuil RMS pour détection de parole (augmenté pour éviter faux positifs)
PREBUFFER_CHUNKS = 3
FADE_OUT_MS = 12
PLAYBACK_ECHO_MARGIN = 0.01  # Marge supplémentaire pour filtrer l'écho


SYSTEM_PROMPT = """You are Samanta, a friendly and professional Luxury Consultant specialized in everything SBM, an expert strategist in luxury hostels and restaurants, 
aware of trends with a creative and marketing mind. Aware of numbers, what works and what doesn’t, with a strong background in brand strategy in advertising 
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
"""





class SamantaRealtimeAgent:
    """Agent vocal temps réel pour les réservations d'hôtels."""
    
    def __init__(self) -> None:
        self.session: RealtimeSession | None = None
        self.audio_stream: sd.InputStream | None = None
        self.audio_player: sd.OutputStream | None = None
        self.recording = False
        self.should_exit = False
        
        # Tracker de lecture audio
        self.playback_tracker = RealtimePlaybackTracker()
        
        # File d'attente audio
        self.output_queue: queue.Queue[Any] = queue.Queue(maxsize=0)
        self.interrupt_event = threading.Event()
        self.current_audio_chunk: tuple[np.ndarray[Any, np.dtype[Any]], str, int] | None = None
        self.chunk_position = 0
        
        # Gestion du jitter buffer et fade-out
        self.prebuffering = True
        self.prebuffer_target_chunks = PREBUFFER_CHUNKS
        self.fading = False
        self.fade_total_samples = 0
        self.fade_done_samples = 0
        self.fade_samples = int(SAMPLE_RATE * (FADE_OUT_MS / 1000.0))
        self.playback_rms = 0.0
        
        # Protection contre auto-interruption au démarrage
        self.warmup_complete = False
        self.warmup_chunks_needed = 200  # 20 chunks = 0.8 secondes de warmup

        self.agent = RealtimeAgent(
            name="Samanta",
            instructions=SYSTEM_PROMPT,
        )


    def _output_callback(self, outdata, frames: int, time, status) -> None:
        """Callback pour la sortie audio."""
        if status:
            print(f"Status sortie audio: {status}")

        # Gestion de l'interruption avec fade-out
        if self.interrupt_event.is_set():
            outdata.fill(0)
            if self.current_audio_chunk is None:
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
                remaining_in_chunk = len(self.current_audio_chunk[0]) - self.chunk_position
                self.fade_total_samples = min(self.fade_samples, max(0, remaining_in_chunk))

            samples, item_id, content_index = self.current_audio_chunk
            samples_filled = 0
            while samples_filled < len(outdata) and self.fade_done_samples < self.fade_total_samples:
                remaining_output = len(outdata) - samples_filled
                remaining_fade = self.fade_total_samples - self.fade_done_samples
                n = min(remaining_output, remaining_fade)
                src = samples[self.chunk_position:self.chunk_position + n].astype(np.float32)
                idx = np.arange(self.fade_done_samples, self.fade_done_samples + n, dtype=np.float32)
                gain = 1.0 - (idx / float(self.fade_total_samples))
                ramped = np.clip(src * gain, -32768.0, 32767.0).astype(np.int16)
                outdata[samples_filled:samples_filled + n, 0] = ramped
                self._update_playback_rms(ramped)
                try:
                    self.playback_tracker.on_play_bytes(item_id=item_id, item_content_index=content_index, bytes=ramped.tobytes())
                except Exception:
                    pass
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

        # Remplissage normal du buffer audio
        outdata.fill(0)
        samples_filled = 0
        while samples_filled < len(outdata):
            if self.current_audio_chunk is None:
                try:
                    if self.prebuffering and self.output_queue.qsize() < self.prebuffer_target_chunks:
                        break
                    self.prebuffering = False
                    self.current_audio_chunk = self.output_queue.get_nowait()
                    self.chunk_position = 0
                except queue.Empty:
                    break

            remaining_output = len(outdata) - samples_filled
            samples, item_id, content_index = self.current_audio_chunk
            remaining_chunk = len(samples) - self.chunk_position
            samples_to_copy = min(remaining_output, remaining_chunk)

            if samples_to_copy > 0:
                chunk_data = samples[self.chunk_position:self.chunk_position + samples_to_copy]
                outdata[samples_filled:samples_filled + samples_to_copy, 0] = chunk_data
                self._update_playback_rms(chunk_data)
                samples_filled += samples_to_copy
                self.chunk_position += samples_to_copy
                try:
                    self.playback_tracker.on_play_bytes(item_id=item_id, item_content_index=content_index, bytes=chunk_data.tobytes())
                except Exception:
                    pass

            if self.chunk_position >= len(samples):
                self.current_audio_chunk = None
                self.chunk_position = 0

    async def run(self) -> None:
        """Lance l'agent Samanta."""
        print("=" * 60)
        print("🏨 Samanta v4 - Agent Vocal Marketing")
        print("=" * 60)
        print("\nCommande Ctrl+C pour quitter")
        print("Connexion en cours...")

        self.audio_player = sd.OutputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=FORMAT,
            callback=self._output_callback,
            blocksize=int(SAMPLE_RATE * CHUNK_LENGTH_S),
        )
        self.audio_player.start()

        try:
            runner = RealtimeRunner(self.agent)
            model_config: RealtimeModelConfig = {
                "playback_tracker": self.playback_tracker,
                "initial_model_settings": {
                    "turn_detection": {
                        "type": "semantic_vad",
                        "interrupt_response": True,
                        "create_response": True,
                    },
                    "voice": "marin",
                },
            }

            async with await runner.run(model_config=model_config) as session:
                self.session = session
                print("✅ Connecté!")

                await self.start_audio_recording()
                print("\nDébut de la conversation")

                async for event in session:
                    await self._on_event(event)

        finally:
            time.sleep(3)
            if self.audio_player and self.audio_player.active:
                self.audio_player.stop()
            if self.audio_player:
                self.audio_player.close()
            print("\n👋 Bye-bye!")

    async def start_audio_recording(self) -> None:
        """Démarre l'enregistrement audio."""
        self.audio_stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=FORMAT,
        )
        self.audio_stream.start()
        self.recording = True
        asyncio.create_task(self.capture_audio())

    async def capture_audio(self) -> None:
        """Capture l'audio du microphone et l'envoie à la session."""
        if not self.audio_stream or not self.session:
            return

        read_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)

        try:
            while self.recording:
                if self.audio_stream.read_available < read_size:
                    await asyncio.sleep(0.01)
                    continue

                data, _ = self.audio_stream.read(read_size)
                audio_bytes = data.tobytes()

                # Détection intelligente de parole pendant la lecture
                assistant_playing = self.current_audio_chunk is not None or not self.output_queue.empty()
                
                # Pendant le warmup, ne pas détecter d'interruptions
                if assistant_playing and not self.warmup_complete:
                    # Pendant le warmup, ne rien envoyer pour éviter auto-interruption
                    pass
                elif assistant_playing:
                    samples = data.reshape(-1)
                    mic_rms = self._compute_rms(samples)
                    # Seuil plus strict: doit être significativement plus fort que l'écho
                    playback_gate = max(ENERGY_THRESHOLD, self.playback_rms * 1.2 + PLAYBACK_ECHO_MARGIN)
                    if mic_rms >= playback_gate:
                        self.interrupt_event.set()
                        await self.session.send_audio(audio_bytes)
                else:
                    await self.session.send_audio(audio_bytes)

                await asyncio.sleep(0)

        except Exception as e:
            # print(f"Erreur capture audio: {e}")
            pass
        finally:
            if self.audio_stream and self.audio_stream.active:
                self.audio_stream.stop()
            if self.audio_stream:
                self.audio_stream.close()

    async def _on_event(self, event: RealtimeSessionEvent) -> None:
        """Gère les événements de la session."""
        try:
            if event.type == "agent_start":
                print("💬 Samanta parle")
            elif event.type == "agent_end":
                # When the agent has finished speaking, check if we need to exit
                if self.should_exit:
                    self.recording = False
                    if self.session:
                        await self.session.close()
            elif event.type == "tool_start":
                print(f"🔧 Outil utilisé: {event.tool.name}")
            elif event.type == "tool_end":
                print(f"✅ Action effectuée")
                # Vérifier si c'est l'outil de fin de conversation
                if event.tool.name == "terminer_conversation":
                    self.should_exit = True
            elif event.type == "audio":
                np_audio = np.frombuffer(event.audio.data, dtype=np.int16)
                self.output_queue.put_nowait((np_audio, event.item_id, event.content_index))
            elif event.type == "audio_interrupted":
                print("🎤 Voix détectée")
                self.prebuffering = True
                self.interrupt_event.set()
            elif event.type == "error":
                print(f"❌ Erreur: {event.error}")
            # elif event.type == "raw_model_event":
            #     # Afficher les transcriptions
            #     data = event.data
                # if data.type != "raw_server_event" and data.type != "transcript_delta" and data.type != "audio":
                #     print(f"type: {data.type}")
                # if hasattr(data, "type"):
                #     if data.type == "turn_started":
                #         print(f"\n💬 Samanta: {data.transcript}")
                #     elif data.type == "conversation.item.input_audio_transcription.completed":
                #         print(f"\n👤 Vous: {data.transcript}")
        except Exception as e:
            print(f"Erreur événement: {str(e)[:100]}")

    def _update_playback_rms(self, samples: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Met à jour l'estimation d'énergie de lecture."""
        sample_rms = self._compute_rms(samples)
        self.playback_rms = 0.9 * self.playback_rms + 0.1 * sample_rms
        
        # Marquer le warmup comme terminé après avoir joué suffisamment de chunks
        if not self.warmup_complete and self.warmup_chunks_needed > 0:
            self.warmup_chunks_needed -= 1
            if self.warmup_chunks_needed == 0:
                self.warmup_complete = True
                # print("🔓 Détection d'interruption activée")
                
    def _compute_rms(self, samples: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Calcule l'énergie RMS des échantillons."""
        if samples.size == 0:
            return 0.0
        x = samples.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x)))



if __name__ == "__main__":
    samanta = SamantaRealtimeAgent()
    try:
        asyncio.run(samanta.run())
    except KeyboardInterrupt:
        sys.exit(0)

