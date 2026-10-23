"""Configuration and settings for the Lecture RAG Tool."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_STORE_DIR = BASE_DIR / "vector_indices"
SAMPLE_DIR = BASE_DIR / "sample_data"

DATA_DIR.mkdir(exist_ok=True)
INDEX_STORE_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AVAILABLE_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "paraphrase-MiniLM-L3-v2"
]

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
MIN_TEXT_CHARS_FOR_OCR = 40  # If page has fewer characters, trigger OCR

SUPPORTED_LLM_PROVIDERS = ["Groq (LLaMA-3)", "Google Gemini", "Ollama (Local)"]

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

@dataclass
class RAGSettings:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = 5
    score_threshold: float = 0.0
    ocr_enabled: bool = True
    llm_provider: str = "Groq (LLaMA-3)"
    llm_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    api_key: str = ""
