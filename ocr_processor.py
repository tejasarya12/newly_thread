"""
OCR Processor: Hybrid Pipeline
Selects between CPU (Tesseract) and GPU (DeepSeek/Ollama) based on user preference.
"""
import logging
import io
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from utils.model_loader import get_ocr_client

logger = logging.getLogger("OCR-Processor")

# CONFIGURATION
OCR_MODEL_NAME = "deepseek-ocr" 

class PDFProcessor:
    def __init__(self):
        self.client = get_ocr_client()

    def extract_text(self, pdf_path, mode='cpu', max_pages=3):
        """
        Main extraction entry point.
        """
        try:
            logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
            logger.info(f"Selected Mode: {mode.upper()}")
            
            # 1. Convert PDF pages to images
            try:
                images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)
            except Exception as e:
                logger.error(f"PDF Conversion Failed. Is Poppler installed? Error: {e}")
                return None
            
            full_text = ""
            
            # 2. Route to specific engine
            for i, image in enumerate(images):
                logger.info(f"   Processing Page {i+1}...")
                
                page_text = ""
                if mode == 'gpu':
                    page_text = self._process_with_deepseek(image)
                else:
                    page_text = self._process_with_tesseract(image)
                
                if page_text:
                    full_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                else:
                    logger.warning(f"   Page {i+1} returned empty text")
                
            return full_text
            
        except Exception as e:
            logger.error(f"OCR Pipeline Failed: {e}")
            return None

    def _process_with_tesseract(self, image: Image.Image):
        """CPU-based OCR using Tesseract"""
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Tesseract Error: {e}")
            return ""

    def _process_with_deepseek(self, image: Image.Image):
        """GPU-based OCR using Ollama Vision Model"""
        try:
            # Convert PIL Image to Byte Array
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            # Call Ollama
            response = self.client.chat(
                model=OCR_MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': 'Transcribe the text in this image exactly as it appears.',
                    'images': [img_bytes]
                }]
            )
            return response.get('message', {}).get('content', '').strip()

        except Exception as e:
            logger.error(f"DeepSeek OCR Error: {e}")
            return ""

def extract_text_from_pdf(filepath, mode='cpu'):
    processor = PDFProcessor()
    return processor.extract_text(filepath, mode=mode)