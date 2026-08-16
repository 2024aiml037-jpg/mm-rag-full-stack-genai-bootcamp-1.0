from __future__ import annotations

import base64
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt_library.prompt import (
    GENERATION_SMOKE_TEST_QUERY,
    MULTIMODAL_RAG_SYSTEM_PROMPT,
    format_multimodal_rag_user_prompt,
    format_retrieved_image_evidence_prompt,
)
from src.retriever import MultimodalQdrantRetriever


logger = CustomLogger().get_logger(__name__)

class MultimodalRAGGenerator:
    def __init__(self):
        pass

    def _validate_config(self):
        # Load the multimodal RAG model here
        pass

    def _image_to_data_url(self) -> str:
        # Convert the image to a data URL
        pass

    def _source_label(self):
        # Implement the generation logic using the model and tokenizer
        pass
    
    def _prepare_context():
        # Prepare the context for generation
        pass
    
    def _build_messages():
        # Build the messages for the RAG model
        pass
    
    def _extract_response_text(self) -> str:
        pass
    
    def _usage_metadata():
        pass
    
    def generate():
        # Implement the generation logic using the model and tokenizer
        pass    
    
    def answer_question():
        pass