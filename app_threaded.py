"""
Main Application - Real-Time Voice AI Assistant
"""
import logging
import signal
import sys
import time
import os
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import chromadb
import pytesseract
from pdf2image import convert_from_path
from ollama import Client

# Lazy Load Embeddings
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = None 
except ImportError:
    embedding_model = None

# Thread imports
from threads.thread_manager import get_thread_manager
from threads.thread1_audio_capture import audio_ingest_thread_main
from threads.thread2_asr import asr_thread_main
from threads.thread3_query_processing import query_processing_thread_main
from threads.thread4_retrieval import retrieval_thread_main
from threads.thread5_llm_generation import llm_generation_thread_main
from ocr_processor import extract_text_from_pdf

from config_threaded import THREADS, MODELS, SYSTEM, QUEUES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("App")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'realtime-secret-key'
app.config['UPLOAD_FOLDER'] = SYSTEM['upload_folder']
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

thread_manager = None

def clear_system_queues():
    """Helper to clear all queues to prevent stale answers"""
    if not thread_manager: return
    
    queues_to_clear = [
        QUEUES["asr_to_process"], 
        QUEUES["process_to_rag"], 
        QUEUES["rag_to_llm"], 
        QUEUES["ui_events"] # Critical for removing old text
    ]
    
    for q_name in queues_to_clear:
        q = thread_manager.get_queue(q_name)
        if q:
            with q.mutex:
                q.queue.clear()
    logger.info("♻️ System Queues Cleared for new input")

def bridge_ui_events():
    global thread_manager
    ui_queue = thread_manager.get_queue(QUEUES["ui_events"])
    logger.info("BRIDGE: UI Event Bridge Started")
    while True:
        try:
            # Faster polling for speed
            data = ui_queue.get(timeout=0.01) 
            if data['type'] == 'transcript':
                socketio.emit('user_transcript', {'text': data['text']})
            elif data['type'] == 'llm_token':
                socketio.emit('ai_response_chunk', {'text': data['text']})
            elif data['type'] == 'status':
                socketio.emit('system_status', {'status': data['text']})
        except:
            continue

def initialize_system():
    global thread_manager
    thread_manager = get_thread_manager()
    
    logger.info("Initializing Real-Time Voice Pipeline")
    
    if THREADS["audio_ingest"]["enabled"]:
        thread_manager.register_thread("audio_ingest", audio_ingest_thread_main, 
                                     queue_in=QUEUES["input_audio"], queue_out=QUEUES["vad_to_asr"])

    if THREADS["asr"]["enabled"]:
        thread_manager.register_thread("asr", asr_thread_main, 
                                     queue_in=QUEUES["vad_to_asr"], queue_out=QUEUES["asr_to_process"])

    if THREADS["query_processing"]["enabled"]:
        thread_manager.register_thread("processing", query_processing_thread_main, 
                                     queue_in=QUEUES["asr_to_process"], queue_out=QUEUES["process_to_rag"])

    if THREADS["retrieval"]["enabled"]:
        thread_manager.register_thread("retrieval", retrieval_thread_main, 
                                     queue_in=QUEUES["process_to_rag"], queue_out=QUEUES["rag_to_llm"])

    if THREADS["llm_generation"]["enabled"]:
        thread_manager.register_thread("llm", llm_generation_thread_main, 
                                     queue_in=QUEUES["rag_to_llm"], queue_out=None)

    thread_manager.start_all()
    bridge_thread = threading.Thread(target=bridge_ui_events, daemon=True)
    bridge_thread.start()
    logger.info("Pipeline Running")

# ROUTES
@app.route('/')
def index():
    return render_template('index_threaded.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/registrations')
def registrations():
    return render_template('registrations.html')

@app.route('/api/companies')
def get_companies():
    try:
        chroma_client = chromadb.PersistentClient(path=SYSTEM['vector_db_path'])
        collection = chroma_client.get_or_create_collection(name="company_documents")
        results = collection.get(include=['metadatas'])
        companies_map = {}
        if results['metadatas']:
            for meta in results['metadatas']:
                c_name = meta.get('company_name', 'Unknown')
                if c_name not in companies_map:
                    companies_map[c_name] = {
                        'name': c_name,
                        'product_name': meta.get('product_name', 'General Product'),
                        'email': meta.get('email', ''),
                        'doc_count': 0,
                        'last_updated': meta.get('upload_date', '')
                    }
                companies_map[c_name]['doc_count'] += 1
                current_date = meta.get('upload_date', '')
                if current_date > companies_map[c_name]['last_updated']:
                    companies_map[c_name]['last_updated'] = current_date
        return jsonify({"status": "success", "companies": list(companies_map.values())})
    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/delete_companies', methods=['POST'])
def delete_companies():
    try:
        data = request.json
        companies_to_delete = data.get('companies', [])
        if not companies_to_delete:
            return jsonify({"status": "error", "message": "No companies selected"}), 400
        chroma_client = chromadb.PersistentClient(path=SYSTEM['vector_db_path'])
        collection = chroma_client.get_or_create_collection(name="company_documents")
        for company in companies_to_delete:
            collection.delete(where={"company_name": company})
        return jsonify({"status": "success", "message": f"Deleted {len(companies_to_delete)} companies"})
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global embedding_model
    try:
        if 'pdf' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
        file = request.files['pdf']
        company_name = request.form.get('company_name', '').strip()
        product_name = request.form.get('product_name', '').strip()
        email = request.form.get('email', '').strip()
        ocr_mode = request.form.get('ocr_mode', 'cpu') 
        
        if not company_name or not product_name:
            return jsonify({"status": "error", "message": "Company/Product Name required"}), 400
            
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{company_name}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        logger.info(f"Processing PDF: {unique_filename} [Mode: {ocr_mode}]")

        full_text = extract_text_from_pdf(filepath, mode=ocr_mode)
        if not full_text or not full_text.strip():
            return jsonify({"status": "error", "message": "OCR found no text"}), 400

        if embedding_model is None:
            logger.info("Loading Sentence Transformer...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence Transformer Loaded")

        try:
            chroma_client = chromadb.PersistentClient(path=SYSTEM['vector_db_path'])
            collection = chroma_client.get_or_create_collection(name="company_documents")
        except Exception as e:
            return jsonify({"status": "error", "message": f"DB Error: {str(e)}"}), 500
        
        chunk_size = 500
        overlap = 50
        step = chunk_size - overlap
        chunks = []
        
        for i in range(0, len(full_text), step):
            chunk_content = full_text[i : i + chunk_size].strip()
            if chunk_content:
                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "source": filename,
                        "company_name": company_name,
                        "product_name": product_name,
                        "email": email,
                        "upload_date": timestamp,
                        "chunk_index": len(chunks)
                    }
                })
        
        ids, docs, metas, embs = [], [], [], []
        logger.info(f"Generating Embeddings for {len(chunks)} chunks...")
        
        batch_size = 8 
        chunk_texts = [c["content"] for c in chunks]
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch).tolist()
            for j, emb in enumerate(batch_embeddings):
                global_idx = i + j
                ids.append(f"{company_name}_{timestamp}_{global_idx}")
                docs.append(chunks[global_idx]["content"])
                metas.append(chunks[global_idx]["metadata"])
                embs.append(emb)
            logger.info(f"   ...embedded batch {i//batch_size + 1}")

        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            logger.info(f"Stored {len(ids)} chunks")
            
        return jsonify({
            "status": "success",
            "message": f"Registered '{product_name}'. Mode: {ocr_mode.upper()}.",
            "chunks": len(ids)
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query', '').strip()
    if not query: return jsonify({"status": "error"}), 400
    
    # CLEAR QUEUES to remove old context/answers
    clear_system_queues()
    
    if thread_manager:
        q = thread_manager.get_queue(QUEUES["asr_to_process"])
        if q:
            q.put({"text": query})
            logger.info(f"Manual Query Injected: {query}")
        else:
            return jsonify({"status": "error", "message": "Pipeline not ready"}), 503
    return jsonify({"status": "success", "answer": "Processing...", "confidence_score": 1.0})

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    emit('server_status', {'ready': True})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('audio_stream')
def handle_audio_stream(data):
    if thread_manager:
        q = thread_manager.get_queue(QUEUES["input_audio"])
        if q:
            q.put(data)

def cleanup(signum, frame):
    logger.info("Shutting down...")
    if thread_manager:
        thread_manager.stop_all()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, cleanup)
    initialize_system()
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
