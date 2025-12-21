"""
utils/model_loader.py
Manages LLM (Text) and VLM (OCR/Vision) connections via Ollama
"""
import logging
from ollama import Client
from config_threaded import MODELS, SYSTEM

logger = logging.getLogger(__name__)

class LocalModelManager:
    """Manages loading and caching of local models via Ollama"""
    
    def __init__(self):
        self.llm_client = None
        self.ocr_client = None
        self.base_url = MODELS['llm']['base_url']
        
    def get_ocr_client(self):
        """Get the Ollama client specifically for Vision/OCR tasks"""
        if self.ocr_client is None:
            try:
                # We use the same client, but this method ensures we validate connection
                self.ocr_client = Client(host=self.base_url)
                logger.info(f"✅ Ollama OCR Client connected at {self.base_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama for OCR: {e}")
        return self.ocr_client

# Global instance
model_manager = LocalModelManager()

def get_ocr_client():
    """Get OCR-specific client"""
    return model_manager.get_ocr_client()