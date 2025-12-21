"""
Thread 1: Audio Ingest & VAD
Receives streaming audio chunks, buffers them, and detects silence.
"""
import logging
import queue
import torch
import numpy as np
from config_threaded import THREADS

logger = logging.getLogger("Thread-1-Ingest")

def audio_ingest_thread_main(manager, input_queue, output_queue, shutdown_event):
    logger.info("Using Silero VAD for voice detection")
    
    # Load Silero VAD
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      trust_repo=True)
        (get_speech_timestamps, _, read_audio, _, _) = utils
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")
        return
    
    # Configuration
    sample_rate = 16000
    # Silero requires EXACTLY 512 samples for 16kHz
    VAD_WINDOW = 512 
    REQUIRED_SILENCE_CHUNKS = 20 # approx 0.6s of silence
    
    # State
    raw_buffer = bytearray() 
    vad_buffer = [] 
    silence_counter = 0
    is_speaking = False
    
    logger.info("VAD Ready. Waiting for audio stream...")

    while not shutdown_event.is_set():
        try:
            # 1. Get raw bytes from WebSocket
            chunk_bytes = input_queue.get(timeout=0.1)
            raw_buffer.extend(chunk_bytes)
            
            # 2. Process in 512-sample chunks
            while len(raw_buffer) >= (VAD_WINDOW * 2):
                chunk = raw_buffer[:(VAD_WINDOW * 2)]
                raw_buffer = raw_buffer[(VAD_WINDOW * 2):]
                
                # Convert to Float32 for VAD
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                # Check VAD
                tensor = torch.from_numpy(audio_float32)
                
                try:
                    speech_prob = model(tensor, sample_rate).item()
                except:
                    continue

                if speech_prob > 0.5:
                    # Speech detected
                    if not is_speaking:
                        logger.info("User started speaking...")
                        is_speaking = True
                    
                    silence_counter = 0
                    vad_buffer.append(chunk)
                else:
                    # Silence detected
                    if is_speaking:
                        silence_counter += 1
                        vad_buffer.append(chunk)
                        
                        if silence_counter > REQUIRED_SILENCE_CHUNKS:
                            logger.info(f"End of speech. Processing {len(vad_buffer)} chunks.")
                            
                            full_audio = b''.join(vad_buffer)
                            output_queue.put(full_audio)
                            
                            # Reset
                            vad_buffer = []
                            is_speaking = False
                            silence_counter = 0
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"VAD Loop Error: {e}")

