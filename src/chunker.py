"""Semantic chunking pipeline tailored for lecture slides and academic PDFs."""
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class LectureChunker:
    """Chunks lecture pages and slides while preserving educational context and metadata."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
            length_function=len
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of lecture documents/slides into contextualized chunks."""
        chunked_docs: List[Document] = []
        chunk_counter = 0

        for doc in documents:
            text = doc.page_content.strip()
            meta = doc.metadata.copy()
            doc_name = meta.get("doc_name", "Unknown Document")
            page_num = meta.get("page", 1)
            slide_title = meta.get("slide_title", f"Slide {page_num}")
            doc_type = meta.get("doc_type", "Document")

            # Context prefix to anchor embedding and LLM comprehension
            header_prefix = f"[{doc_type}: {doc_name} | Slide/Page {page_num}: {slide_title}]\n"

            # If slide text is within standard chunk size, preserve it as a single coherent unit
            if len(text) <= self.chunk_size:
                chunk_counter += 1
                meta["chunk_id"] = f"{doc_name}_p{page_num}_c1"
                meta["chunk_index"] = chunk_counter
                meta["citation"] = f"{doc_name} (Page/Slide {page_num})"
                # Inject contextual header into chunk content
                enriched_content = f"{header_prefix}{text}"
                chunked_docs.append(Document(page_content=enriched_content, metadata=meta))
            else:
                # Text exceeds chunk size; use recursive character splitting
                splits = self.text_splitter.split_text(text)
                for sub_idx, split_text in enumerate(splits):
                    chunk_counter += 1
                    sub_meta = meta.copy()
                    sub_meta["chunk_id"] = f"{doc_name}_p{page_num}_c{sub_idx + 1}"
                    sub_meta["chunk_index"] = chunk_counter
                    sub_meta["sub_chunk"] = sub_idx + 1
                    sub_meta["total_sub_chunks"] = len(splits)
                    sub_meta["citation"] = f"{doc_name} (Page/Slide {page_num}, Part {sub_idx + 1})"
                    
                    enriched_content = f"{header_prefix}{split_text}"
                    chunked_docs.append(Document(page_content=enriched_content, metadata=sub_meta))

        logger.info(f"Chunked {len(documents)} source pages into {len(chunked_docs)} semantic chunks.")
        return chunked_docs
