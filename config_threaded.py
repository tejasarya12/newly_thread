"""
Configuration for Real-Time Voice RAG Architecture
"""
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM = {
    "log_level": "INFO",
    "upload_folder": "uploads",
    "vector_db_path": "vector_db",
    "device": "cpu", #cuda
}

THREADS = {
    "audio_ingest": {
        "enabled": True,
        "sample_rate": 16000,
        "vad_model": "silero",
        "silence_threshold": 0.5,
        "chunk_size": 512,
    },
    "asr": {
        "enabled": True,
        "model_size": "tiny.en",
        "compute_type": "int8",
        "beam_size": 1,
    },
    "query_processing": {
        "enabled": True,
        "intent_detection": False,
        "remove_filler_words": True,
    },
    "retrieval": {
        "enabled": True,
        "top_k": 3,
        "score_threshold": 0.3,
        "use_reranker": False,
    },
    "llm_generation": {
        "enabled": True,
        "streaming": True,
        "max_tokens": 150,
        "temperature": 0.3,
    },
    "audio_playback": {
        "enabled": True,
        "sample_rate": 22050,
        "buffer_size": 1024,
        "volume": 1.0,
        "interrupt_enabled": True
    }
}

MODELS = {
    "llm": {
        "provider": "ollama",
        "model_name": "tinyllama",
        "base_url": "http://localhost:11434",
    },
    "embedding": {
        "provider": "sentence-transformers",
        "model_name": "all-MiniLM-L6-v2"
    },
    "tts": {
        "enabled": True,
        "model": "piper",
        "model_path": "models/en_US-lessac-medium.onnx", 
        "config_path": "models/en_US-lessac-medium.onnx.json",
    }
}

QUEUES = {
    "input_audio": "q_audio_ingest",
    "vad_to_asr": "q_asr_input",
    "asr_to_process": "q_process_input",
    "process_to_rag": "q_rag_input",
    "rag_to_llm": "q_llm_input",
    "llm_to_tts": "q_tts_input", 
    "tts_to_play": "q_audio_playback",
    "ui_events": "q_ui_events"
}