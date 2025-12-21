<h1>🎙️ Real-Time Voice RAG Assistant </h1>
A high-performance, voice-to-voice AI support agent that runs entirely locally. This project implements a multi-threaded pipeline to capture voice, transcribe it, retrieve relevant context from uploaded documentation (RAG), and generate spoken responses in real-time.

<h1>⚡ Key Features </h1>
Real-Time Voice Pipeline: Seamless voice interaction with low-latency streaming (ASR → LLM → TTS).

Local RAG (Retrieval-Augmented Generation): Upload PDF documentation to create a custom knowledge base. The AI answers strictly based on your data.

Hybrid OCR Engine: Toggle between CPU (Tesseract) and GPU (DeepSeek-VL/Ollama) text extraction for uploaded documents.

Multi-Threaded Architecture: Optimized Python backend using 7 concurrent threads for Audio Ingest, Transcription, Retrieval, and Generation to ensure non-blocking performance.

Modern UI: specialized "Cinematic" frontend with dark/light mode, magnetic UI elements, and real-time transcription updates via WebSockets.

<h1>🛠️ Tech Stack</h1>
Core: Python, Flask, Socket.IO

AI & ML: Ollama (Llama 3), Faster-Whisper, Sentence-Transformers, ChromaDB

Frontend: Tailwind CSS, GSAP Animations, Web Speech API

<h1>1.directory path structure :</h1>

in ai_orch_thread folder {
in langraph_flow : __init__.py , graph_build.py,nodes.py
in templates : all html files
in threads : all thread ( 7 ) files , __init__.py
uploads : empty folder
in utils : __init.py__ , model_loader.py
vector_db : empty folder


app_threaded.py
config_threaded.py
ocr_processor.py
}


<h1>2.To exicute :</h1>

download ollama desktop
exicute in cmd :
ollama pull tinyllama (or any suitable model , tinyllama is light weight so i have choosen it )
ollama run tinyllama ( to check if model has been installed successfully)
/bye (to exit)

NOTE : download only if you have a gpu support (refer deepseek-ocr webpage for the spec needed )
ollama pull deepseek-ocr ( to get deepseek ocr and to perform better extraction of information from pdf uploaded )

in vscode terminal :

python -m venv venv
.\venv\Scripts\activate 
pip install -r requirements.txt

 cd "d:\ai orch thread"; .\venv\Scripts\activate; cd "ai-orch-thread"; python .\config_threaded.py
 cd "d:\ai orch thread"; .\venv\Scripts\activate; cd "ai-orch-thread"; python app_threaded.py   
