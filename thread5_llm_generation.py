"""
Thread 5: LLM Generation
"""
import logging
import queue
from ollama import Client
from config_threaded import THREADS, MODELS, QUEUES

logger = logging.getLogger("Thread-5-LLM")

def llm_generation_thread_main(manager, input_queue, output_queue, shutdown_event):
    client = Client(host=MODELS['llm']['base_url'])
    model_name = MODELS['llm']['model_name']
    
    # Get UI Queue to stream text to frontend
    ui_queue = manager.get_queue(QUEUES["ui_events"])
    
    logger.info(f"LLM Thread Ready (Model: {model_name})")
    
    while not shutdown_event.is_set():
        try:
            data = input_queue.get(timeout=0.1)
            query = data.get("original", "")
            context = data.get("context", "")
            
            # Prevent Hallucination Prompt
            if context:
                system_prompt = (


                    "You are a helpful assistant. Use the provided Context to answer the user's Question. "
                    "If the answer isn't perfect, just give the best summary you can based on the Context.\n"
                    f"Context:\n{context}"
                )
            else:
                # If no docs found, be honest
                system_prompt = "You are a product support AI. No relevant documents were found for this query. Please state that you don't have the information."
            
            stream = client.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': query}
                ],
                stream=True,
            )
            
            logger.info("Generating response...")
            
            sentence_buffer = ""
            
            for chunk in stream:
                token = chunk['message']['content']
                sentence_buffer += token
                
                # UPDATE UI (Stream token)
                if ui_queue:
                    ui_queue.put({"type": "llm_token", "text": token})
                
                # Send to TTS queue ONLY if output_queue exists
                # (In current config, output_queue is None because we use browser TTS)
                if output_queue:
                    if any(punct in token for punct in ['.', '!', '?', '\n']):
                        if sentence_buffer.strip():
                            output_queue.put(sentence_buffer)
                            sentence_buffer = ""
            
            # Flush remaining buffer if queue exists
            if output_queue and sentence_buffer.strip():
                output_queue.put(sentence_buffer)
                output_queue.put("<END_OF_TURN>")
            
        except queue.Empty:
            continue
        except Exception as e:

            logger.error(f"LLM Error: {e}")
