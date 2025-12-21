# newly_thread:

<h1>1.directory location :</h1>

in ai_orch_thread folder {
in langraph_flow : __init__.py , graph_build.py,nodes.py
in templates : all html files
in threads : all thread ( 7 ) files , __init__.py
uploads : empty folder
in utils : __init.py__ , model_loader.py
vector_db : empty folder

download_vosk_model.py
app_threaded.py
config_threaded.py
ocr_processor.py
}


<h1>2.to run exicute :</h1>

download ollama desktop
exicute in cmd :
ollama pull tinyllama (or any suitable model , tinyllama is light weight so i have choosen it )
ollama run tinyllama ( to check if model has been installed successfully)
/bye (to exit)

NOTE : download only if you have a gpu support (refer deepseek-ocr webpage for the spec needed )
ollama pull deepseek-ocr ( to get deepseek ocr )



python -m venv venv
.\venv\Scripts\activate 
pip install -r requirements.txt

run download_vosk_model.py to get a proper voice intake.



 cd "d:\ai orch thread"; .\venv\Scripts\activate; cd "ai-orch-thread"; python .\config_threaded.py
 cd "d:\ai orch thread"; .\venv\Scripts\activate; cd "ai-orch-thread"; python app_threaded.py   
