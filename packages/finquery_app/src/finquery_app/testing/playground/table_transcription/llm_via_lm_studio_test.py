import requests
import base64
import json
from pdf2image import convert_from_path
import io
import fitz
from operator import itemgetter

MODEL_NAME = "gemma-3n-e4b-it-mlx@4bit"
PDF_FILE_PATH = "../../test_docs/aapl-20230930-short.pdf"
LM_STUDIO_URL = "http://localhost:12345/v1/chat/completions"


def extract_structured_text_from_pdf(filepath):
    try:
        print("Extracting structured text from PDF...")
        doc = fitz.open(filepath)
        page = doc.load_page(0)

        words = page.get_text("words")
        doc.close()

        if not words:
            return ""

        lines = {}
        for w in words:
            line_key = round(w[3])
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(w)

        sorted_lines = sorted(lines.items(), key=itemgetter(0))

        reconstructed_text = ""
        for _, line_words in sorted_lines:
            line_words.sort(key=itemgetter(0))
            reconstructed_text += " ".join(w[4] for w in line_words) + "\n"

        return reconstructed_text
    except Exception as e:
        print(f"Error extracting structured text with PyMuPDF: {e}")
        return None


def convert_pdf_to_high_res_image_base64(filepath):
    try:
        print("Converting PDF to high-resolution image (300 DPI)...")
        images = convert_from_path(filepath, dpi=300, first_page=1, last_page=1)

        if not images:
            print("Error: Could not convert PDF to image.")
            return None

        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        print("Encoding image to base64...")
        return base64.b64encode(img_byte_arr).decode('utf-8')

    except Exception as e:
        print(f"Error during PDF conversion: {e}")
        print("Please ensure 'poppler' is installed and accessible in your system's PATH.")
        return None


def process_pdf_with_lm_studio():
    structured_pdf_text = extract_structured_text_from_pdf(PDF_FILE_PATH)
    if not structured_pdf_text:
        return

    base64_image = convert_pdf_to_high_res_image_base64(PDF_FILE_PATH)
    if not base64_image:
        return

    PROMPT = f"""
You are an expert financial data processor. Your task is to create a perfect, clean markdown table from the provided materials.

**Your Strategy is Critical:**
1.  **Primary Source for Data:** The "STRUCTURED TEXT" provided below is your single source of truth for all labels, numbers, and the CORRECT VERTICAL ORDER OF ROWS.
2.  **Primary Source for Layout:** Use the **IMAGE** primarily to understand the column structure and final Markdown formatting. Do NOT use it for OCR.
3.  **Combine Sources:** Reconstruct the table by mapping the data from the structured text onto the layout you see in the image.

**General Rules for High Accuracy:**
-   Your entire response must be ONLY the final markdown table. Do not include any introductory text, summaries, or ` ```markdown ` markers.
-   Meticulously preserve all financial formatting: `$` signs, commas, and parentheses `()` for negative values.
-   **Ambiguity Rule:** If a line in the structured text seems complex or contains unusual formatting (e.g., multiple labels, slashes, or mid-line parentheses), treat it as a single, complete row. Do not skip it or merge it with other rows. Your goal is to represent every line from the structured text in the final table.

---BEGIN STRUCTURED TEXT (Row by Row)---
{structured_pdf_text}
---END STRUCTURED TEXT---
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1
    }

    try:
        response = requests.post(LM_STUDIO_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

        print("\n--- Generated Markdown Table ---")
        print(content)
        print("\n------------------------------\n")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Details: {e}")


if __name__ == "__main__":
    process_pdf_with_lm_studio()

"""
**Apple Inc.**
**CONSOLIDATED STATEMENTS OF OPERATIONS**
(In millions, except number of shares, which are reflected in thousands, and per-share amounts)

|                                     | **Year ended September 30, 2023** | **Year ended September 30, 2022** | **Year ended September 30, 2021** |
|-------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| **Net sales:**                       |                                    |                                    |                                    |
| Products                             | $ 298,085                          | $ 316,199                          | $ 297,392                          |
| Services                            | $ 85,200                           | $ 78,129                           | $ 68,425                           |
| **Total net sales**                  | **$ 383,285**                      | **$ 394,328**                      | **$ 365,817**                      |
| **Cost of sales:**                   |                                    |                                    |                                    |
| Products                             | $ 189,282                          | $ 201,471                          | $ 192,266                          |
| Services                            | $ 24,855                           | $ 22,075                           | $ 20,715                           |
| **Total cost of sales**              | **$ 214,137**                      | **$ 223,546**                      | **$ 212,981**                      |
| **Gross margin**                     | **$ 169,148**                      | **$ 170,782**                      | **$ 152,836**                      |
| **Operating expenses:**               |                                    |                                    |                                    |
| Research and development              | $ 29,915                           | $ 26,251                           | $ 21,914                           |
| Selling, general and administrative   | $ 24,932                           | $ 25,094                           | $ 21,973                           |
| **Total operating expenses**          | **$ 54,847**                       | **$ 51,345**                       | **$ 43,887**                       |
| **Operating income (loss), net**      | **$ 114,301**                      | **$ 119,437**                      | **$ 108,949**                      |
| Other income (expense), net           | $(565)                            | $(334)                            | $ 258                             |
| Income before provision for income taxes | $ 113,736                          | $ 119,103                          | $ 109,207                          |
| Provision for income taxes            | $ 16,741                           | $ 19,300                           | $ 14,527                           |
| **Net income**                        | **$ 96,995**                       | **$ 99,803**                       | **$ 94,680**                       |
| **Earnings per share:**               |                                    |                                    |                                    |
| Basic                                | $ 6.16                             | $ 6.15                             | $ 5.67                             |
| Diluted                              | $ 6.13                             | $ 6.11                             | $ 5.61                             |
| **Shares used in computing earnings per share:** |                                    |                                    |                                    |
| Basic                                | 15,744,231                        | 16,215,963                        | 16,701,272                        |
| Diluted                              | 15,812,547                        | 16,325,819                        | 16,864,919                        |
| **See accompanying Notes to Consolidated Financial Statements.** |                                    |                                    |                                    |
| Apple Inc. © 2023 Form 10-K | 28 |
"""