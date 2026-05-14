
import io
from pathlib import Path
import numpy as np
import easyocr
import pymupdf
import pytesseract
from PIL import Image

def ocr_image(image: Image.Image) -> str:
      """Run OCR on a PIL image and return extracted text using easyocr."""
      reader = easyocr.Reader(['en'], gpu=False) 
      image_array = np.array(image)
      result = reader.readtext(image_array, detail=0, paragraph=True)
      return "\n".join(result).strip()



def upscale_if_small(img: Image.Image, min_dim: int = 1500) -> Image.Image:
    """Upscale tiny images so OCR has enough resolution."""
    w, h = img.size
    if max(w, h) < min_dim:
        scale = min_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def extract_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes, using OCR for pages with little text."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)
        else:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            pages.append(ocr_image(upscale_if_small(img)))
    return "\n\n---\n\n".join(pages)


def extract_image(image_bytes: bytes) -> str:
    """Extract text from image bytes using OCR."""
    img = Image.open(io.BytesIO(image_bytes))
    return ocr_image(upscale_if_small(img))


def extract_raw_text(file_bytes: bytes, filename: str) -> str:
    
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(file_bytes)
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        return extract_image(file_bytes)
    raise ValueError(f"Unsupported file type: '{ext}'")
