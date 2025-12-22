"""
Thread 6: TTS (Kokoro)
High-quality, Human-like speech synthesis
"""
import logging
import queue
import io
import wave
import numpy as np
import soundfile as sf

# Try importing Kokoro
try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

from config_threaded import THREADS, MODELS

logger = logging.getLogger("Thread-6-TTS")

class KokoroTTSThread:
    def __init__(self, input_queue, output_queue, shutdown_event):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event
        
        self.model_path = MODELS["tts"]["model_path"]
        self.voices_path = MODELS["tts"]["voices_path"]
        self.voice_name = MODELS["tts"]["voice_name"]
        self.kokoro = None
        
    def load_model(self):
        if not KOKORO_AVAILABLE:
            logger.error("Kokoro not installed. Run: pip install kokoro-onnx soundfile")
            return

        try:
            logger.info(f"Loading Kokoro TTS: {self.model_path}")
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            logger.info(f"✅ Kokoro Ready (Voice: {self.voice_name})")
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            self.kokoro = None

    def synthesize_stream(self, text):
        if not self.kokoro:
            return

        try:
            # Generate Audio (Returns raw samples, sample_rate)
            samples, sample_rate = self.kokoro.create_audio(
                text, 
                self.voice_name, 
                speed=1.0, 
                lang="en-us"
            )
            
            # Convert float32 numpy array to int16 PCM bytes for playback
            # Scale -1.0 to 1.0 -> -32767 to 32767
            audio_int16 = (samples * 32767).astype(np.int16)
            
            # Create WAV container in memory
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2) # 2 bytes for int16
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                audio_data = wav_buffer.getvalue()
                
                # Send to playback thread
                self.output_queue.put({
                    "type": "audio_chunk",
                    "data": audio_data
                })
                logger.info(f"Synthesized: '{text[:20]}...'")

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")

    def run(self):
        self.load_model()
        
        while not self.shutdown_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
                
                if item == "<END_OF_TURN>":
                    continue 
                
                if isinstance(item, str) and item.strip():
                    # Filter out purely non-verbal strings to save time
                    if any(c.isalnum() for c in item):
                        self.synthesize_stream(item)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS Error: {e}")

def tts_thread_main(manager, input_queue, output_queue, shutdown_event):
    thread = KokoroTTSThread(
        input_queue=input_queue,
        output_queue=output_queue,
        shutdown_event=shutdown_event
    )
    thread.run()





"""
Thread 6: Streaming TTS (Piper)
Converts LLM text stream into audio bytes immediately.

import logging
import queue
import wave
import io
import numpy as np
try:
    from piper import PiperVoice 
except ImportError:
    logging.error("Piper TTS not installed. Run: pip install piper-tts")

from config_threaded import THREADS, MODELS

logger = logging.getLogger("Thread-6-TTS")

class PiperTTSThread:
    def __init__(self, input_queue, output_queue, shutdown_event):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event
        
        # Load Config
        self.model_path = MODELS["tts"]["model_path"]
        self.config_path = MODELS["tts"]["config_path"]
        self.voice = None
        
    def load_model(self):
        try:
            logger.info(f"Loading Piper Model: {self.model_path}")
            self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)
            logger.info("Piper TTS Loaded")
        except Exception as e:
            logger.error(f"Failed to load Piper: {e}")
            self.voice = None

    def synthesize_stream(self, text):
        if not self.voice:
            return

        # Create an in-memory buffer for the audio
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wav_file:
                # Piper synthesizes to a wav file object
                self.voice.synthesize(text, wav_file)
            
            # Get the bytes
            audio_data = wav_buffer.getvalue()
            
            # Send to playback thread
            self.output_queue.put({
                "type": "audio_chunk",
                "data": audio_data
            })

    def run(self):
        self.load_model()
        
        while not self.shutdown_event.is_set():
            try:
                # Get text chunk from LLM
                item = self.input_queue.get(timeout=0.1)
                
                if item == "<END_OF_TURN>":
                    continue 
                
                if isinstance(item, str) and item.strip():
                    # Synthesize
                    self.synthesize_stream(item)
                    logger.info(f"Synthesized: '{item}'")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS Error: {e}")

# FIXED: Accepts 4 arguments to match ThreadManager
def tts_thread_main(manager, input_queue, output_queue, shutdown_event):
    thread = PiperTTSThread(
        input_queue=input_queue,
        output_queue=output_queue,
        shutdown_event=shutdown_event
    )
    thread.run()






"""
Thread 6: TTS (Text-to-Speech) - Windows Safe
Replaces Coqui TTS with pyttsx3 for offline speech synthesis

import logging
import pyttsx3
import tempfile
import wave
import numpy as np
from queue import Empty
from config_threaded import THREADS

logger = logging.getLogger(__name__)


class TTSThread:
   

    def __init__(self, input_queue, output_queue, shutdown_event):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event

        # Load config
        config = THREADS.get("tts", {})
        self.rate = config.get("rate", 175)
        self.volume = config.get("volume", 1.0)

        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", self.rate)
        self.engine.setProperty("volume", self.volume)

        self.sample_rate = 22050
        logger.info("TTSThread initialized (pyttsx3)")

    def synthesize_text(self, text: str) -> np.ndarray:
       
        if not text.strip():
            return np.array([], dtype=np.float32)

        try:
            # Temporary WAV file
            fd, path = tempfile.mkstemp(suffix=".wav")
            path = path.replace("\\", "/")
            import os
            os.close(fd)

            # Generate speech
            self.engine.save_to_file(text, path)
            self.engine.runAndWait()

            # Load WAV
            with wave.open(path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0

            # Clean up temp file
            try:
                os.remove(path)
            except:
                pass

            return audio

        except Exception as e:
            logger.error(f"pyttsx3 TTS error: {e}")
            return np.array([], dtype=np.float32)

    def process_llm_output(self, llm_data):
        
        text = ""
        if isinstance(llm_data, dict):
            # Handle streaming tokens later if needed
            text = llm_data.get("answer") or llm_data.get("text", "")
        elif isinstance(llm_data, str):
            text = llm_data

        if not text.strip():
            return

        audio = self.synthesize_text(text)
        if len(audio) > 0:
            try:
                self.output_queue.put({
                    "type": "audio_complete",
                    "audio": audio,
                    "sample_rate": self.sample_rate,
                    "text": text,
                    "is_final": True
                }, block=True, timeout=5)
                logger.info(f"TTS audio sent: {len(audio)/self.sample_rate:.2f}s")
            except Exception as e:
                logger.error(f"Failed to send TTS audio: {e}")

    def run(self):
        
        logger.info("TTS thread started")

        while not self.shutdown_event.is_set():
            try:
                llm_data = self.input_queue.get(timeout=0.1)
                self.process_llm_output(llm_data)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"TTS thread error: {e}")

        logger.info("TTS thread stopping...")

    def cleanup(self):
      
        try:
            self.engine.stop()
        except:
            pass
        logger.info("TTS thread cleaned up")


def tts_thread_main(manager):
    
    thread = TTSThread(
        input_queue=manager.get_queue("llm_output"),
        output_queue=manager.get_queue("tts_audio"),
        shutdown_event=manager.shutdown_event
    )
    thread.run()



"""
