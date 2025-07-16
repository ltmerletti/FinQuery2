import os
import argparse
import traceback
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, LayoutOptions
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_LARGE, LayoutModelConfig
from docling.datamodel.base_models import InputFormat

DEFAULT_PDF_PATH = "../../test_docs/tsla-20230930.pdf"

def extract_table_with_docling(filepath):
    if not os.path.exists(filepath):
        print(f"Error: The file was not found at the specified path: '{filepath}'")
        return

    print(f"Processing '{filepath}' with docling using high-accuracy settings...")

    try:
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.table_structure_options.do_cell_matching = False
        # pipeline_options.layout_options = LayoutOptions(
        #     model_spec=DOCLING_LAYOUT_EGRET_LARGE
        # )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(filepath)
        doc = result.document

        if not doc or not hasattr(doc, 'tables') or not doc.tables:
            print("\n--- PROCESSING COMPLETE ---")
            print("Docling processed the file, but no tables were found in the document.")
            return

        print(f"\nFound {len(doc.tables)} table(s) in the document.")

        for i, table in enumerate(doc.tables):
            page_num = table.prov[0].page_no if table.prov else 'Unknown'
            print(f"\n--- Table {i + 1} (from Page {page_num}) ---")

            markdown_output = table.export_to_markdown(doc=doc)

            print(markdown_output)
            print("-" * (len(f"Table {i + 1}...") + 20))

    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print("----------------------")
        print("\nStrategic Advice: This error is unusual. It might indicate a problem with docling's underlying")
        print("dependencies (like PyTorch, MLX, or Poppler) or the environment itself.")
        print("Please ensure all dependencies are correctly installed and the PDF file is not corrupted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tables from a PDF using docling with high-accuracy settings."
    )
    parser.add_argument(
        "filepath",
        nargs='?',
        default=DEFAULT_PDF_PATH,
        help=f"Path to the PDF file to process. Defaults to '{DEFAULT_PDF_PATH}'"
    )
    args = parser.parse_args()

    extract_table_with_docling(args.filepath)


"""
--- Table 1 (from Page 1) ---
|                                              | Years ended        | Years ended        | Years ended        |
|----------------------------------------------|--------------------|--------------------|--------------------|
|                                              | September 30, 2023 | September 24, 2022 | September 25, 2021 |
| Net sales:                                   |                    |                    |                    |
| Products                                     | $ 298,085          | $ 316,199          | $ 297,392          |
| Services                                     | 85,200             | 78,129             | 68,425             |
| Total net sales                              | 383,285            | 394,328            | 365,817            |
| Cost of sales:                               |                    |                    |                    |
| Products                                     | 189,282            | 201,471            | 192,266            |
| Services                                     | 24,855             | 22,075             | 20,715             |
| Total cost of sales                          | 214,137            | 223,546            | 212,981            |
| Gross margin                                 | 169,148            | 170,782            | 152,836            |
| Operating expenses:                          |                    |                    |                    |
| Research and development                     | 29,915             | 26,251             | 21,914             |
| Selling, general and administrative          | 24,932             | 25,094             | 21,973             |
| Total operating expenses                     | 54,847             | 51,345             | 43,887             |
| Operating income                             | 114,301            | 119,437            | 108,949            |
| Other income/(expense), net                  | (565)              | (334)              | 258                |
| Income before provision for income taxes     | 113,736            | 119,103            | 109,207            |
| Provision for income taxes                   | 16,741             | 19,300             | 14,527             |
| Net income                                   | 96,995             | $ 99,803           | $ 94,680           |
| Earnings per share:                          |                    |                    |                    |
| Basic                                        | $ 6.16             | $ 6.15             | $ 5.67             |
| Diluted                                      | 6.13               | $ 6.11             | $ 5.61             |
| Shares used in computing earnings per share: |                    |                    |                    |
| Basic                                        | 15,744,231         | 16,215,963         | 16,701,272         |
| Diluted                                      | 15,812,547         | 16,325,819         | 16,864,919         |
------------------------------
"""