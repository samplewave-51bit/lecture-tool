"""OCR Engine for extracting text from scanned slides, diagrams, and images."""
import io
import logging
from typing import Optional, List, Dict, Any, Union
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class OCREngine:
    """Wrapper around OCR backends (RapidOCR ONNX with Tesseract fallback)."""
    
    _instance = None
    _rapid_ocr = None

    def __init__(self):
        self._init_ocr()

    def _init_ocr(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._rapid_ocr = RapidOCR()
            logger.info("RapidOCR (ONNX) initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize RapidOCR: {e}")
            self._rapid_ocr = None

    def extract_text_from_image(self, image: Union[Image.Image, bytes, np.ndarray]) -> str:
        """Extract text from an image (PIL Image, bytes, or numpy array)."""
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        if isinstance(image, Image.Image):
            # Convert to RGB numpy array
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_np = np.array(image)
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            return ""

        extracted_lines = []
        if self._rapid_ocr is not None:
            try:
                ocr_result, elapse = self._rapid_ocr(img_np)
                if ocr_result:
                    # Sort results by vertical position (y coordinate) then horizontal (x coordinate)
                    # ocr_result item format: [box_coordinates, text, confidence]
                    sorted_boxes = sorted(
                        ocr_result, 
                        key=lambda item: (round(item[0][0][1] / 15.0) * 15, item[0][0][0])
                    )
                    for item in sorted_boxes:
                        text = item[1].strip()
                        conf = item[2]
                        if text and conf > 0.4:
                            extracted_lines.append(text)
                    return "\n".join(extracted_lines)
            except Exception as e:
                logger.error(f"Error during RapidOCR execution: {e}")

        # Fallback to pytesseract if installed and available
        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception:
            pass

        return "\n".join(extracted_lines)


# Global singleton instance
_engine: Optional[OCREngine] = None

def get_ocr_engine() -> OCREngine:
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine
