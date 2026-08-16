import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
from langchain_core.documents import Document

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger


logger = CustomLogger().get_logger(__name__)



class ComplexPDFParser:
    def __init__(self, pdf_path):
        pass
    
    def _validate_pdf(self):
        pass
    
    def _create_output_directory(self):
        pass
    
    def _configure_tesseract(self):
        pass
    
    def run_ocr_on_images(self):
        pass
    
    def extract_text_and_images (self):
        pass
    
    def extract_tables(self):
        pass
    
    def create_langchain_documents(self):
        pass
    
    def _save_json(self):
        pass
    
    def save_output(self):
        pass
    
    def parse(self):
        pass