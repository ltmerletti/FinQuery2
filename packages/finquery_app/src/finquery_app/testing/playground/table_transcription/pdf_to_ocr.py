import os
import fitz  # PyMuPDF
import pymupdf4llm

# --- Tesseract Configuration ---
# Set the path to your Tesseract data directory.
# This path was confirmed in our previous conversation.
os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/opt/tesseract/share/tessdata'


def ocr_and_extract_with_llm(source_filepath: str):
    """
    Performs OCR on an entire PDF document and extracts its content
    into Markdown format using PyMuPDF4LLM.

    Args:
        source_filepath: The path to the source PDF file.
    """
    if not os.path.exists(source_filepath):
        print(f"Error: Source file not found at '{source_filepath}'")
        return

    ocr_pdf_path = "temp_ocr_output.pdf"

    try:
        print(f"--- Phase 1: Performing OCR on '{source_filepath}' ---")

        source_doc = fitz.open(source_filepath)
        ocr_doc = fitz.open()  # Create a new, empty PDF to store OCR'd pages

        # Iterate through each page of the source document
        for i, page in enumerate(source_doc):
            print(f"Processing page {i + 1}/{len(source_doc)}...")

            # 1. Render page to an image (Pixmap)
            pix = page.get_pixmap(dpi=300)

            # 2. OCR the image and get the result as a 1-page PDF in memory
            page_ocr_bytes = pix.pdfocr_tobytes()  #

            # 3. Open the 1-page OCR'd PDF from memory
            page_ocr_doc = fitz.open("pdf", page_ocr_bytes)

            # 4. Insert this OCR'd page into our main output document
            ocr_doc.insert_pdf(page_ocr_doc)
            page_ocr_doc.close()

        print(f"\nSaving fully OCR'd document to '{ocr_pdf_path}'...")
        ocr_doc.save(ocr_pdf_path)
        source_doc.close()
        ocr_doc.close()

        print(f"\n--- Phase 2: Extracting text with PyMuPDF4LLM ---")

        # 5. Use pymupdf4llm.to_markdown to process the searchable PDF
        md_output = pymupdf4llm.to_markdown(ocr_pdf_path)

        # 6. Print the final Markdown content
        print("\n--- Final Markdown Output ---")
        print(md_output)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 7. Clean up the temporary OCR'd PDF file
        if os.path.exists(ocr_pdf_path):
            os.remove(ocr_pdf_path)
            print(f"\nCleaned up temporary file: '{ocr_pdf_path}'")


# --- Main execution ---
if __name__ == "__main__":
    # Replace this with the path to your actual PDF file
    source_pdf = "/Users/lukem/PycharmProjects/FinQuery2/reports/pltr-20231231.pdf"
    ocr_and_extract_with_llm(source_pdf)