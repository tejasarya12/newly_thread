"""
Thread 2: ASR (Faster Whisper)
"""
import logging
import queue
import io
import wave
from faster_whisper import WhisperModel
from config_threaded import THREADS, SYSTEM, QUEUES

logger = logging.getLogger("Thread-2-ASR")

def create_wav_header(pcm_data, sample_rate=16000, channels=1, bits_per_sample=16):
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(bits_per_sample // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()

def asr_thread_main(manager, input_queue, output_queue, shutdown_event):
    config = THREADS['asr']
    device = SYSTEM['device']
    
    # Get UI Queue
    ui_queue = manager.get_queue(QUEUES["ui_events"])
    
    logger.info(f"Loading Faster-Whisper ({config['model_size']}) on {device}...")
    try:
        model = WhisperModel(config['model_size'], device=device, compute_type=config['compute_type'])
        logger.info("Faster-Whisper Ready")
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        return
    
    while not shutdown_event.is_set():
        try:
            audio_bytes = input_queue.get(timeout=0.1)
            
            if not audio_bytes or len(audio_bytes) < 32000:
                continue

            wav_data = create_wav_header(audio_bytes)
            audio_file = io.BytesIO(wav_data)
            
            segments, info = model.transcribe(audio_file, beam_size=config['beam_size'])
            text_segments = [segment.text for segment in segments]
            final_text = " ".join(text_segments).strip()
            
            if final_text:
                logger.info(f"Transcribed: '{final_text}'")
                output_queue.put({"text": final_text})
                
                # UPDATE UI
                if ui_queue:
                    ui_queue.put({"type": "transcript", "text": final_text})
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"ASR Error: {e}")