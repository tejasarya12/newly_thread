"""
Thread 3: Query Processing
"""
import logging
import queue
from config_threaded import THREADS

logger = logging.getLogger("Thread-3-Process")

def query_processing_thread_main(manager, input_queue, output_queue, shutdown_event):
    while not shutdown_event.is_set():
        try:
            data = input_queue.get(timeout=0.1)
            raw_text = data.get("text", "")
            
            if not raw_text:
                continue
                
            cleaned_text = raw_text.lower().strip()
            
            logger.info(f"Processing Query: {cleaned_text}")
            
            processed_data = {
                "original": raw_text,
                "cleaned": cleaned_text,
            }
            
            output_queue.put(processed_data)
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Processing Error: {e}")