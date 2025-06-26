import pathlib
import re
from typing import List

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    # regex function to clean element text
    def _clean_element_text(self, text: str) -> str:
        # removes urls (we don't need them for the type of query)
        text = re.sub(r'https?://\S+', '', text, flags=re.MULTILINE)
        # remove sec.gov footer information
        text = re.sub(r'\S*www\.sec\.gov\S*', '', text, flags=re.MULTILINE)
        # remove page number (which is in format (a number)/(a number)
        text = re.sub(r'\s*\d+/\d+\s*', '', text)
        # removes dates and times
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}(,\s*\d{1,2}:\d{2}\s*(AM|PM)?)?', '', text)
        # removes stray page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # removes empty lines
        text = re.sub(r'\n\s*\n', '\n', text)

        return text.strip()

    def _load_pdf(self, pdf_file_path: pathlib.Path) -> List[Document]:
        print(f"Partitioning document: {pdf_file_path}")

        if not str(pdf_file_path).endswith(".pdf"):
            return []

        elements = list(partition_pdf(pdf_file_path, strategy="hi_res", infer_table_structure=True))

        print("\n--- Found Elements ---")

        final_chunks = []

        company_ticker = pdf_file_path.stem.split('-')[0].upper()

        table_elements = []
        text_elements = []

        # split text and table elements
        # this way we can run targeted postprocessing on the tables and texts for better quality
        for element in elements:
            if isinstance(element, Table):
                table_elements.append(element)
            else:
                text_elements.append(element)

        print(f"--- Separated content into {len(table_elements)} tables and {len(text_elements)} text elements. ---")

        # run the clean text on non tables
        for table_el in table_elements:
            table_text = table_el.metadata.text_as_html if hasattr(table_el.metadata,
                                                                   'text_as_html') and table_el.metadata.text_as_html else table_el.text

            cleaned_text = self._clean_element_text(table_text)

            new_chunk = Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_file_path.name,
                    "page": table_el.metadata.page_number,
                    "company": company_ticker,
                    "element_type": "table"
                }
            )

            final_chunks.append(new_chunk)

        text_chunks = chunk_by_title(
            text_elements,
            max_characters=1000,
            new_after_n_chars=800,
            combine_text_under_n_chars=500,
            overlap=200,
            overlap_all=True
        )

        for chunk in text_chunks:
            cleaned_text = self._clean_element_text(chunk.text)

            if len(cleaned_text) > 50:
                new_chunk = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": pdf_file_path.name,
                        "page": chunk.metadata.page_number,
                        "company": company_ticker,
                        "element_type": "text"
                    }
                )
                final_chunks.append(new_chunk)

        print(f"--- Finished processing. Generated {len(final_chunks)} total chunks for {pdf_file_path.name}. ---\n")

        return final_chunks

    def load(self) -> List[Document]:
        return self._load_pdf(self.file_path)
