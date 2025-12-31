"""
Samanta v3 - Agent vocal temps réel pour réservations d'hôtels de luxe
Utilise l'API Realtime d'OpenAI pour une interaction vocale à faible latence.
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
ENERGY_THRESHOLD = 0.05  # Seuil RMS pour détection de parole (augmenté pour éviter faux positifs)
PREBUFFER_CHUNKS = 3
FADE_OUT_MS = 12
PLAYBACK_ECHO_MARGIN = 0.01  # Marge supplémentaire pour filtrer l'écho


def load_database(path:str) -> str:
    """Charge la base de données d'un hôtel."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, path)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()



# Définition des outils pour l'agent

@function_tool
def rechercher_hotel(nom_hotel: str) -> str:
    """Recherche les informations sur un hôtel spécifique.
        Prend en entrée le nom de l'hôtel."""
    nom_lower = nom_hotel.lower()
    if "royal" in nom_lower or "palace" in nom_lower or "paris" in nom_lower:
        return load_database("database/hotels/paris.txt")
    elif "azur" in nom_lower or "nice" in nom_lower or "méditerranée" in nom_lower:
        return load_database("database/hotels/nice.txt")
    elif "mont" in nom_lower or "blanc" in nom_lower or "chamonix" in nom_lower:
        return load_database("database/hotels/chamonix.txt")
    return "Hôtel non trouvé."


@function_tool
def reserver_hotel(nom_hotel: str, nom_client: str, date_arrivee: str, date_depart: str, nombre_personnes: int) -> str:
    """Réserve un hôtel.
        Prend en entrée le nom de l'hôtel, le nom du client, la date d'arrivée, la date de départ et le nombre de personnes."""
    csv_file = "database/reservations.csv"
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["nom_hotel", "nom_client", "date_arrivee", "date_depart", "nombre_personnes"])
        writer.writerow([nom_hotel, nom_client, date_arrivee, date_depart, nombre_personnes])
    
    return f"Hôtel {nom_hotel} réservé du {date_arrivee} au {date_depart} pour {nombre_personnes} personnes."

@function_tool
def annuler_reservation(nom_hotel: str, nom_client: str, date_arrivee: str, date_depart: str, nombre_personnes: int) -> str:
    """Annule une réservation.
        Prend en entrée le nom de l'hôtel, le nom du client, la date d'arrivée, la date de départ et le nombre de personnes."""
    csv_file = "database/reservations.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["nom_hotel", "nom_client", "date_arrivee", "date_depart", "nombre_personnes"])
        writer.writerow([nom_hotel, nom_client, date_arrivee, date_depart, nombre_personnes])
    return f"Réservation de l'hôtel {nom_hotel} pour {nom_client} du {date_arrivee} au {date_depart} pour {nombre_personnes} personnes annulée."

@function_tool
def obtenir_informations_reservation(nom_client: str) -> str:
    """Obtient les informations d'une réservation.
        Prend en entrée le nom du client."""
    csv_file = "database/reservations.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "r", newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == nom_client:
                return f"Réservation de l'hôtel {row[0]} pour {row[1]} du {row[2]} au {row[3]} pour {row[4]} personnes."

@function_tool
def terminer_conversation() -> str:
    """Termine la conversation."""
    return "Conversation terminée"  





# Instructions système pour l'agent
SYSTEM_INSTRUCTIONS = f"""Tu es Samanta, une assistante vocale chaleureuse et professionnelle spécialisée dans les réservations d'hôtels de luxe SBM.

Ton rôle:
- Répondre aux questions des clients sur les hôtels, chambres, restaurants et activités.
- Aider à la réservation et fournir des recommandations personnalisées.
- Si le client veut connaître des informations sur un hôtel, utilise la fonction 'rechercher_hotel'.
- Si le client veut réserver un hôtel, utilise la fonction 'reserver_hotel'. Demande lui les informations nécessaires.
- Si le client veut annuler une réservation, utilise la fonction 'annuler_reservation'.
- Si le client veut obtenir les informations d'une réservation, utilise la fonction 'obtenir_informations_reservation'.
- Si le client veut terminer la conversation, dis à l'oral 'Au revoir et merci pour votre visite!' puis utilise la fonction 'terminer_conversation'.
- Être concise mais informative (tes réponses seront lues à voix haute).
- Toujours répondre en français avec un ton élégant et accueillant.

Informations complémentaires:
{load_database("database/infos_generales.txt")}

Les hôtels de la société SBM:
1. Hôtel Royal Palace - Paris (5 étoiles, vue Tour Eiffel)
2. Hôtel Côte d'Azur Prestige - Nice (4 étoiles, plage privée)
3. Hôtel Mont-Blanc Excellence - Chamonix (5 étoiles, ski et montagne)

Instructions importantes:
- Commence par te présenter chaleureusement lors du premier échange.
- Utilise le vouvoiement, soit poli. 
- Réponds de manière naturelle et conversationnelle.
- Évite les listes à puces, préfère des phrases fluides.
- Si tu ne trouves pas l'information demandée, ne les invente pas et dis que tu ne disposes pas de ces informations.
- Garde tes réponses relativement courtes (2-4 phrases) sauf si plus de détails sont demandés.
"""


phrase_accueil = "Bonjour et bienvenue! Je suis Samanta, votre assistante personnelle pour les réservations d'hôtels SBM. Comment puis-je vous aider aujourd'hui?"



# Configuration de l'agent
agent = RealtimeAgent(
    name="Samanta",
    instructions=SYSTEM_INSTRUCTIONS,
    tools=[rechercher_hotel, reserver_hotel, annuler_reservation, obtenir_informations_reservation, terminer_conversation],
)



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
        print("🏨 Samanta v3 - Agent Vocal SBM")
        print("=" * 60)
        print("\nCommande Ctrl+C pour quitter")
        print("Connexion en cours...")

        chunk_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)
        self.audio_player = sd.OutputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype=FORMAT,
            callback=self._output_callback,
            blocksize=chunk_size,
        )
        self.audio_player.start()

        try:
            runner = RealtimeRunner(agent)
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
                
                # Démarrer la conversation avec un message d'accueil
                await session.send_message(f"Enonce la phrase d'accueil: {phrase_accueil}")

                await self.start_audio_recording()
                print("\nDébut de la conversation")

                async for event in session:
                    await self._on_event(event)

        finally:
            if self.audio_player and self.audio_player.active:
                self.audio_player.stop()
            if self.audio_player:
                self.audio_player.close()
            print("\n👋 Session terminée")

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
            print("")
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
                # Si nous devons quitter après que l'agent ait fini de parler
                if self.should_exit:
                    print("\n\n👋 Conversation terminée par l'utilisateur")
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
            elif event.type == "audio_end":
                pass  # Silencieux
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
        print("\n\n👋 Au revoir!")
        sys.exit(0)

