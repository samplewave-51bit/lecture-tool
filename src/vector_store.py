"""FAISS Vector Store manager powered by SentenceTransformers embeddings."""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path

import numpy as np
import transformers

# Suppress HuggingFace / PyTorch weight loading progress bars and warnings permanently
transformers.utils.logging.disable_progress_bar()
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import DEFAULT_EMBEDDING_MODEL, INDEX_STORE_DIR

logger = logging.getLogger(__name__)

# Global singleton cache for embedding models to prevent reloading weights on Streamlit reruns
_MODEL_CACHE: Dict[Tuple[str, str], SentenceTransformer] = {}

def get_cached_sentence_transformer(model_name: str, device: str = "cpu") -> SentenceTransformer:
    """Retrieve or load a cached SentenceTransformer instance with progress bars disabled."""
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        logger.info(f"Loading SentenceTransformer '{model_name}' on {device} into memory cache...")
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper for SentenceTransformer embedding models with caching."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.client = get_cached_sentence_transformer(model_name=model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs with normalization for cosine similarity."""
        if not texts:
            return []
        embeddings = self.client.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text with normalization."""
        embedding = self.client.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()


class LectureVectorStore:
    """Manages indexing, multi-document aggregation, and vector similarity search with FAISS."""

    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.embedding_model_name = embedding_model_name
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self.faiss_index: Optional[FAISS] = None
        self.documents_metadata: Dict[str, Dict[str, Any]] = {}  # {doc_name: {chunks: int, pages: int, type: str}}

    @property
    def is_empty(self) -> bool:
        return self.faiss_index is None

    @property
    def total_chunks(self) -> int:
        if self.faiss_index is None:
            return 0
        return self.faiss_index.index.ntotal

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Return list of indexed lecture documents with summary stats."""
        result = []
        for doc_name, meta in self.documents_metadata.items():
            result.append({
                "doc_name": doc_name,
                "chunks": meta.get("chunks", 0),
                "pages": meta.get("pages", 0),
                "doc_type": meta.get("doc_type", "Unknown"),
                "ocr_applied": meta.get("ocr_applied", False)
            })
        return result

    def add_documents(self, documents: List[Document], doc_name: str, doc_type: str = "PDF", total_pages: int = 1) -> int:
        """Add semantic chunks for a document to the FAISS index."""
        if not documents:
            return 0

        # Check if any chunk had OCR
        ocr_applied = any(doc.metadata.get("ocr_applied", False) for doc in documents)

        if self.faiss_index is None:
            self.faiss_index = FAISS.from_documents(documents, self.embeddings)
        else:
            self.faiss_index.add_documents(documents)

        # Update metadata registry
        current_doc_meta = self.documents_metadata.get(doc_name, {"chunks": 0, "pages": total_pages, "doc_type": doc_type, "ocr_applied": False})
        current_doc_meta["chunks"] = current_doc_meta.get("chunks", 0) + len(documents)
        current_doc_meta["pages"] = max(current_doc_meta.get("pages", 1), total_pages)
        current_doc_meta["doc_type"] = doc_type
        current_doc_meta["ocr_applied"] = current_doc_meta.get("ocr_applied", False) or ocr_applied
        self.documents_metadata[doc_name] = current_doc_meta

        logger.info(f"Added {len(documents)} chunks from '{doc_name}'. Total index chunks: {self.total_chunks}")
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_doc_names: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform vector similarity search with optional multi-document filtering."""
        if self.faiss_index is None:
            return []

        # In FAISS, similarity_search_with_score returns L2 distance or negative inner product
        # When embeddings are normalized, inner product is cosine similarity
        results_with_score = self.faiss_index.similarity_search_with_score(
            query=query,
            k=k * 2 if filter_doc_names else k  # Fetch extra to filter if needed
        )

        filtered_results: List[Tuple[Document, float]] = []
        for doc, score in results_with_score:
            doc_name = doc.metadata.get("doc_name")
            if filter_doc_names and doc_name not in filter_doc_names:
                continue

            # Convert distance to normalized similarity score (0 to 1)
            # For normalized vectors with L2 distance: dist^2 = 2 - 2*cos_sim => cos_sim = 1 - dist^2/2
            similarity = max(0.0, min(1.0, 1.0 - (score / 2.0))) if score >= 0 else 1.0 / (1.0 + abs(score))
            filtered_results.append((doc, similarity))

            if len(filtered_results) >= k:
                break

        return filtered_results

    def save_to_disk(self, folder_path: Optional[Path] = None, index_name: str = "lecture_index"):
        """Save vector store and document metadata to disk."""
        target_dir = folder_path or (INDEX_STORE_DIR / index_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.faiss_index is not None:
            self.faiss_index.save_local(str(target_dir))
            # Save document metadata registry
            meta_path = target_dir / "documents_registry.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.documents_metadata, f, indent=2)
            logger.info(f"Vector store successfully saved to {target_dir}")

    def load_from_disk(self, folder_path: Optional[Path] = None, index_name: str = "lecture_index") -> bool:
        """Load vector store and document metadata from disk."""
        target_dir = folder_path or (INDEX_STORE_DIR / index_name)
        if not (target_dir / "index.faiss").exists():
            return False

        try:
            self.faiss_index = FAISS.load_local(
                str(target_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            meta_path = target_dir / "documents_registry.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.documents_metadata = json.load(f)
            logger.info(f"Loaded vector store from {target_dir} with {self.total_chunks} chunks.")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store from {target_dir}: {e}")
            return False

    def clear(self):
        """Reset the vector store."""
        self.faiss_index = None
        self.documents_metadata.clear()
        logger.info("Vector store reset.")
