import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_XLARGE
from docling.datamodel.pipeline_options import (LayoutOptions, PdfPipelineOptions, TableFormerMode,
                                                TableStructureOptions, RapidOcrOptions)
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- Configuration ---
# Set the Tesseract data path for PyMuPDF's OCR function
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/opt/tesseract/share/tessdata'

# Define file paths for the pipeline
SOURCE_PDF = "/Users/lukem/PycharmProjects/FinQuery2/reports/pltr-20231231.pdf"
PHASE1_OUTPUT_PDF = "/Users/lukem/PycharmProjects/FinQuery2/reports/preprocessed_output.pdf"
PHASE2_OUTPUT_PDF = "/Users/lukem/PycharmProjects/FinQuery2/reports/final_ocr_output.pdf"


# --- Helper Functions for Preprocessing ---
def deskew_image(image: np.ndarray) -> np.ndarray:
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = osd['rotate']
        if angle != 0:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            return deskewed
    except pytesseract.TesseractError as e:
        print(f"  - Could not determine skew: {e}. Skipping deskew.")
    return image

def denoise_image(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

def binarize_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)


# --- Pipeline Phase Functions ---
def phase1_preprocess_pdf(source_filepath: str, output_filepath: str) -> bool:
    """Takes a raw PDF, cleans each page, and saves a new PDF."""
    print(f"--- Phase 1: Preprocessing PDF ---")
    print(f"Input: {source_filepath}")
    try:
        images = convert_from_path(source_filepath, dpi=300)
        processed_pil_images = []
        for i, pil_image in enumerate(images):
            print(f"  - Cleaning page {i + 1}/{len(images)}...")
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            deskewed = deskew_image(cv_image)
            denoised = denoise_image(deskewed)
            binarized = binarize_image(denoised)
            final_pil_image = Image.fromarray(cv2.cvtColor(binarized, cv2.COLOR_BGR2RGB))
            processed_pil_images.append(final_pil_image)

        if processed_pil_images:
            processed_pil_images[0].save(output_filepath, save_all=True, append_images=processed_pil_images[1:])
            print(f"Success! Saved preprocessed PDF to: {output_filepath}")
            return True
    except Exception as e:
        print(f"Error in Phase 1: {e}")
        return False

def phase2_ocr_with_pymupdf(source_filepath: str, output_filepath: str) -> bool:
    """Takes a preprocessed PDF, performs OCR, and saves a new searchable PDF."""
    print(f"\n--- Phase 2: Performing OCR with PyMuPDF ---")
    print(f"Input: {source_filepath}")
    try:
        source_doc = fitz.open(source_filepath)
        ocr_doc = fitz.open()
        for i, page in enumerate(source_doc):
            print(f"  - OCR'ing page {i + 1}/{len(source_doc)}...")
            pix = page.get_pixmap(dpi=600)
            page_ocr_bytes = pix.pdfocr_tobytes()
            page_ocr_doc = fitz.open("pdf", page_ocr_bytes)
            ocr_doc.insert_pdf(page_ocr_doc)
            page_ocr_doc.close()

        ocr_doc.save(output_filepath)
        print(f"Success! Saved searchable PDF to: {output_filepath}")
        source_doc.close()
        ocr_doc.close()
        return True
    except Exception as e:
        print(f"Error in Phase 2: {e}")
        return False

def phase3_parse_with_docling(filepath: str):
    """Takes a fully OCR'd PDF and parses it with Docling."""
    print(f"\n--- Phase 3: Parsing with Docling ---")
    print(f"Input: {filepath}")
    try:
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE, do_cell_matching=False),
            layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_XLARGE),
            images_scale=2.0,
            generate_page_images=True,
            accelerator_options=AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CPU),
            ocr_options=RapidOcrOptions(lang=["en"], force_full_page_ocr=True)
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(filepath)
        doc = result.document.export_to_markdown()
        print("\n--- FINAL DOCLING MARKDOWN OUTPUT ---")
        print(doc)
        print("--- END OF DOCLING OUTPUT ---")
    except Exception as e:
        print(f"Error in Phase 3: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        p1_ok = phase1_preprocess_pdf(SOURCE_PDF, PHASE1_OUTPUT_PDF)
        if p1_ok:
            p2_ok = phase2_ocr_with_pymupdf(PHASE1_OUTPUT_PDF, PHASE2_OUTPUT_PDF)
            if p2_ok:
                phase3_parse_with_docling(PHASE2_OUTPUT_PDF)
    finally:
        print("\n--- Cleaning up intermediate files ---")
        for f in [PHASE1_OUTPUT_PDF, PHASE2_OUTPUT_PDF]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed: {f}")