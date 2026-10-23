"""Document ingestion and extraction pipelines using PyMuPDF and python-pptx with OCR fallback."""
import io
import os
import re
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image

import pymupdf  # PyMuPDF 2.0+
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from langchain_core.documents import Document

from src.ocr_engine import get_ocr_engine
from src.config import MIN_TEXT_CHARS_FOR_OCR

logger = logging.getLogger(__name__)

def clean_lecture_text(text: str) -> str:
    """Clean and normalize extracted lecture text."""
    if not text:
        return ""
    # Normalize unicode spaces and repeated line breaks
    text = re.sub(r'\r\n|\r', '\n', text)
    # Remove excessive consecutive empty lines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove strange non-printable characters except standard whitespace
    text = re.sub(r'[^\x20-\x7E\n\t\u00A0-\u024F\u0370-\u03FF\u2010-\u2026\u2200-\u22FF]', ' ', text)
    # Collapse consecutive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


class LectureParser:
    """Unified parser for PDFs and PPT/PPTX files with OCR fallback."""

    def __init__(self, ocr_enabled: bool = True, min_chars_for_ocr: int = MIN_TEXT_CHARS_FOR_OCR):
        self.ocr_enabled = ocr_enabled
        self.min_chars_for_ocr = min_chars_for_ocr
        self.ocr_engine = get_ocr_engine() if ocr_enabled else None

    def parse_file(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Parse a PDF or PPTX file from a filepath or file-like stream."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return self.parse_pdf(file_source, filename)
        elif ext in [".pptx", ".ppt"]:
            return self.parse_pptx(file_source, filename)
        elif ext in [".txt", ".md"]:
            return self.parse_text_file(file_source, filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: PDF, PPTX, TXT, MD.")

    def parse_pdf(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Extract pages from PDF using PyMuPDF with automatic OCR on scanned pages."""
        docs: List[Document] = []
        
        if isinstance(file_source, (str, Path)):
            doc = pymupdf.open(str(file_source))
        else:
            # BytesIO or bytes
            file_bytes = file_source.getvalue() if hasattr(file_source, "getvalue") else file_source.read()
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")

        total_pages = len(doc)
        logger.info(f"Parsing PDF '{filename}' with {total_pages} pages.")

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            ocr_applied = False
            
            # 1. Try native text extraction
            text = page.get_text("text")
            cleaned_text = clean_lecture_text(text)
            
            # Detect slide title (usually top block)
            title = ""
            blocks = page.get_text("blocks")
            if blocks and len(blocks) > 0:
                first_block_text = blocks[0][4].strip().replace('\n', ' ')
                if len(first_block_text) < 120:
                    title = first_block_text

            # 2. Check if page appears scanned or empty (< threshold characters)
            if len(cleaned_text) < self.min_chars_for_ocr and self.ocr_enabled:
                logger.info(f"Page {page_num} of {filename} has low text ({len(cleaned_text)} chars). Triggering OCR.")
                try:
                    # Render page to pixmap
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    ocr_text = self.ocr_engine.extract_text_from_image(img_bytes) if self.ocr_engine else ""
                    if len(ocr_text) > len(cleaned_text):
                        cleaned_text = clean_lecture_text(ocr_text)
                        ocr_applied = True
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num} of {filename}: {e}")

            if not cleaned_text:
                cleaned_text = f"[Empty or Graphical Slide {page_num}]"

            # Create standard Document
            metadata = {
                "source": filename,
                "doc_name": filename,
                "page": page_num,
                "total_pages": total_pages,
                "slide_title": title or f"Slide {page_num}",
                "doc_type": "PDF",
                "ocr_applied": ocr_applied
            }
            docs.append(Document(page_content=cleaned_text, metadata=metadata))

        doc.close()
        return docs

    def parse_pptx(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Extract slides, titles, notes, and shape text from PPTX files with OCR on image-heavy slides."""
        docs: List[Document] = []
        
        if isinstance(file_source, (str, Path)):
            prs = Presentation(str(file_source))
        else:
            prs = Presentation(file_source)

        total_slides = len(prs.slides)
        logger.info(f"Parsing PPTX '{filename}' with {total_slides} slides.")

        for slide_idx, slide in enumerate(prs.slides):
            slide_num = slide_idx + 1
            ocr_applied = False
            slide_text_elements = []
            slide_title = ""
            image_blobs = []

            # 1. Slide Title
            if slide.shapes.title and slide.shapes.title.has_text_frame:
                slide_title = slide.shapes.title.text_frame.text.strip()
                if slide_title:
                    slide_text_elements.append(f"Title: {slide_title}")

            # 2. Iterate Shapes for text, tables, and images
            for shape in slide.shapes:
                # Text frames
                if shape.has_text_frame and shape != slide.shapes.title:
                    shape_text = shape.text_frame.text.strip()
                    if shape_text:
                        slide_text_elements.append(shape_text)
                
                # Tables
                elif shape.has_table:
                    table_rows = []
                    for row in shape.table.rows:
                        row_cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                        table_rows.append(" | ".join(row_cells))
                    if table_rows:
                        slide_text_elements.append("Table Data:\n" + "\n".join(table_rows))
                
                # Extract image for OCR if shape is an image
                elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE and hasattr(shape, "image"):
                    try:
                        image_blobs.append(shape.image.blob)
                    except Exception:
                        pass

            # 3. Speaker Notes
            notes_text = ""
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    notes_text = notes
                    slide_text_elements.append(f"Speaker Notes: {notes}")

            combined_text = "\n\n".join(slide_text_elements)
            cleaned_text = clean_lecture_text(combined_text)

            # 4. OCR on slide images if text is minimal
            if len(cleaned_text) < self.min_chars_for_ocr and image_blobs and self.ocr_enabled and self.ocr_engine:
                logger.info(f"Slide {slide_num} of {filename} has low text. Running OCR on {len(image_blobs)} images.")
                ocr_results = []
                for img_blob in image_blobs:
                    try:
                        txt = self.ocr_engine.extract_text_from_image(img_blob)
                        if txt:
                            ocr_results.append(txt)
                    except Exception as e:
                        logger.warning(f"Error doing OCR on PPTX slide {slide_num} image: {e}")
                
                if ocr_results:
                    extra_ocr = "\n".join(ocr_results)
                    cleaned_text = clean_lecture_text(cleaned_text + "\n\n[OCR from Slide Diagrams]:\n" + extra_ocr)
                    ocr_applied = True

            if not cleaned_text:
                cleaned_text = f"[Visual / Title Slide {slide_num}: {slide_title or 'Untitled'}]"

            metadata = {
                "source": filename,
                "doc_name": filename,
                "page": slide_num,
                "total_pages": total_slides,
                "slide_title": slide_title or f"Slide {slide_num}",
                "doc_type": "PPTX",
                "ocr_applied": ocr_applied,
                "has_notes": bool(notes_text)
            }
            docs.append(Document(page_content=cleaned_text, metadata=metadata))

        return docs

    def parse_text_file(self, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[Document]:
        """Parse raw text or markdown lecture notes."""
        if isinstance(file_source, (str, Path)):
            with open(str(file_source), "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            content = file_source.read().decode("utf-8", errors="ignore")
        
        cleaned = clean_lecture_text(content)
        return [
            Document(
                page_content=cleaned,
                metadata={
                    "source": filename,
                    "doc_name": filename,
                    "page": 1,
                    "total_pages": 1,
                    "slide_title": Path(filename).stem,
                    "doc_type": "TEXT",
                    "ocr_applied": False
                }
            )
        ]
