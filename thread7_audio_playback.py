"""
Thread 7: Audio Playback (Streaming)
Plays audio chunks received from TTS. Handles interruptions.
"""
import logging
import queue
import pyaudio
import io
import wave

logger = logging.getLogger("Thread-7-Play")

class AudioPlaybackThread:
    def __init__(self, input_queue, shutdown_event):
        self.input_queue = input_queue
        self.shutdown_event = shutdown_event
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False

    def play_chunk(self, wav_bytes):
        """Plays a single WAV chunk"""
        try:
            # Parse WAV header to get format
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                # Open stream if format changed or not open
                if self.stream is None: 
                    self.stream = self.p.open(
                        format=self.p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True
                    )
                
                # Read frames
                chunk = 1024
                data = wf.readframes(chunk)
                
                self.is_playing = True
                while data:
                    if self.shutdown_event.is_set():
                        break
                    
                    # Check for interrupt signal
                    # (Note: In a complex system, you'd check a separate queue/event here)
                    
                    self.stream.write(data)
                    data = wf.readframes(chunk)
                    
                self.is_playing = False
                
        except Exception as e:
            logger.error(f"Playback Error: {e}")

    def run(self):
        logger.info("Audio Playback Ready")
        
        while not self.shutdown_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
                
                if isinstance(item, dict) and item.get("type") == "interrupt":
                    logger.info("Playback Interrupted")
                    if self.stream:
                        self.stream.stop_stream()
                        self.stream.close()
                        self.stream = None
                    # Clear queue
                    with self.input_queue.mutex:
                        self.input_queue.queue.clear()
                    continue
                
                if isinstance(item, dict) and item.get("type") == "audio_chunk":
                    self.play_chunk(item["data"])
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback Thread Error: {e}")
        
        self.cleanup()

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

# FIXED: Accepts 4 arguments. output_queue will be None, which is fine.
def audio_playback_thread_main(manager, input_queue, output_queue, shutdown_event):
    thread = AudioPlaybackThread(
        input_queue=input_queue,
        shutdown_event=shutdown_event
    )
    thread.run()












"""
Thread 7: Audio Playback
Plays synthesized speech with interrupt handling

import logging
import numpy as np
import pyaudio
from queue import Queue, Empty
from threading import Event, Lock
from config_threaded import THREADS

logger = logging.getLogger(__name__)


class AudioPlaybackThread:
   
    
    def __init__(self, input_queue: Queue, shutdown_event: Event):
        self.input_queue = input_queue
        self.shutdown_event = shutdown_event
        
        # Configuration
        config = THREADS["audio_playback"]
        self.sample_rate = config["sample_rate"]
        self.buffer_size = config["buffer_size"]
        self.volume = config["volume"]
        self.interrupt_enabled = config["interrupt_enabled"]
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Playback control
        self.is_playing = False
        self.playback_lock = Lock()
        self.interrupt_event = Event()
        
        # Buffer
        self.audio_buffer = []
        
        logger.info("AudioPlaybackThread initialized")
    
    def setup_audio_stream(self, sample_rate: int):
       
        try:
            # Close existing stream if any
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            # Open new stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size
            )
            
            self.sample_rate = sample_rate
            logger.info(f"Audio stream opened: {sample_rate}Hz")
            
        except Exception as e:
            logger.error(f"Failed to setup audio stream: {e}")
            raise
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
       
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Peak normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Apply volume
        audio = audio * self.volume
        
        # Clip to [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def resample_audio(self, audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
      
        if source_rate == target_rate:
            return audio
        
        try:
            from scipy import signal
            
            # Calculate resampling ratio
            num_samples = int(len(audio) * target_rate / source_rate)
            
            # Resample
            resampled = signal.resample(audio, num_samples)
            
            return resampled.astype(np.float32)
        
        except ImportError:
            logger.warning("scipy not available, skipping resampling")
            return audio
    
    def play_audio_chunk(self, audio_chunk: dict):
        
        try:
            audio = audio_chunk["audio"]
            source_rate = audio_chunk["sample_rate"]
            chunk_index = audio_chunk.get("chunk_index", 0)
            is_final = audio_chunk.get("is_final", False)
            
            # Setup stream if needed
            if not self.stream or self.sample_rate != source_rate:
                self.setup_audio_stream(source_rate)
            
            # Normalize audio
            audio = self.normalize_audio(audio)
            
            # Check for interrupt before playing
            if self.interrupt_event.is_set():
                logger.info("Playback interrupted before chunk")
                return
            
            with self.playback_lock:
                self.is_playing = True
                
                # Play audio
                logger.debug(f"Playing chunk {chunk_index}: {len(audio)} samples")
                
                # Write audio in chunks to allow interruption
                chunk_size = self.buffer_size
                for i in range(0, len(audio), chunk_size):
                    # Check for interrupt
                    if self.interrupt_event.is_set():
                        logger.info(f"Playback interrupted at chunk {chunk_index}")
                        break
                    
                    # Get chunk
                    chunk = audio[i:i + chunk_size]
                    
                    # Write to stream
                    self.stream.write(chunk.tobytes())
                
                self.is_playing = False
            
            if is_final:
                logger.info("Final audio chunk played")
        
        except Exception as e:
            logger.error(f"Error playing audio chunk: {e}")
            self.is_playing = False
    
    def play_complete_audio(self, audio_data: dict):
        
        try:
            audio = audio_data["audio"]
            source_rate = audio_data["sample_rate"]
            
            # Setup stream
            if not self.stream or self.sample_rate != source_rate:
                self.setup_audio_stream(source_rate)
            
            # Normalize
            audio = self.normalize_audio(audio)
            
            with self.playback_lock:
                self.is_playing = True
                
                logger.info(f"Playing complete audio: {len(audio)} samples, {len(audio)/source_rate:.2f}s")
                
                # Play in chunks to allow interruption
                chunk_size = self.buffer_size
                for i in range(0, len(audio), chunk_size):
                    if self.interrupt_event.is_set():
                        logger.info("Playback interrupted")
                        break
                    
                    chunk = audio[i:i + chunk_size]
                    self.stream.write(chunk.tobytes())
                
                self.is_playing = False
                logger.info("Playback complete")
        
        except Exception as e:
            logger.error(f"Error playing complete audio: {e}")
            self.is_playing = False
    
    def stop_playback(self):
        
        if self.is_playing:
            logger.info("Stopping playback...")
            self.interrupt_event.set()
            
            # Wait for playback to stop
            with self.playback_lock:
                # Clear stream buffer
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.start_stream()
            
            self.interrupt_event.clear()
            logger.info("Playback stopped")
    
    def handle_interrupt(self):
        
        if self.interrupt_enabled and self.is_playing:
            logger.info("🔴 Barge-in detected - interrupting playback")
            self.stop_playback()
            
            # Clear remaining audio in queue
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except:
                    break
    
    def run(self):
        
        logger.info("Audio playback thread started")
        
        try:
            # Process audio chunks
            while not self.shutdown_event.is_set():
                try:
                    # Get audio data from TTS queue
                    audio_data = self.input_queue.get(timeout=0.1)
                    
                    data_type = audio_data.get("type")
                    
                    if data_type == "audio_chunk":
                        # Streaming chunk
                        self.play_audio_chunk(audio_data)
                    
                    elif data_type == "audio_complete":
                        # Complete audio
                        self.play_complete_audio(audio_data)
                    
                    elif data_type == "interrupt":
                        # Interrupt signal
                        self.handle_interrupt()
                
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in audio playback: {e}", exc_info=True)
            
            logger.info("Audio playback thread stopping...")
        
        except Exception as e:
            logger.error(f"Audio playback thread error: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
       
        # Stop playback
        if self.is_playing:
            self.stop_playback()
        
        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Terminate PyAudio
        if self.audio:
            self.audio.terminate()
        
        logger.info("Audio playback thread cleaned up")


def audio_playback_thread_main(manager):
   
    thread = AudioPlaybackThread(
        input_queue=manager.get_queue("tts_audio"),
        shutdown_event=manager.shutdown_event
    )
    thread.run()



"""