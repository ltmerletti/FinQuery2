import pathlib
import re
from typing import List, Optional, Tuple

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

class Context:
    def __init__(
        self,
        pdf_title: str = "",
        page_number: int = 0,
        section_title: str = "",
        relevant_keywords: list[str] = None,
        summary: Optional[str] = None
    ):
        self.pdf_title = pdf_title
        self.page_number = page_number
        self.section_title = section_title
        self.relevant_keywords = relevant_keywords or []
        self.summary = summary or None

    def _to_string(self):
        return f"""
    [CONTEXT]
    PDF Title: {self.pdf_title}
    Page Number: {self.page_number}
    Section Title: {self.section_title}
    Relevant Keywords: {str(self.relevant_keywords)}
    """

def _clean_element_text(text: str) -> str:
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


def getSectionTitle(param):
    pass


def getRelevantKeywords():
    pass


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def _partition_and_separate_elements(self, pdf_file_path: pathlib.Path) -> Tuple[List, List]:
        """
        Partitions the PDF and separates its elements into tables and text.
        """
        print(f"Partitioning document: {pdf_file_path}")

        if not str(pdf_file_path).endswith(".pdf"):
            return [], []

        elements = list(partition_pdf(pdf_file_path, strategy="hi_res", infer_table_structure=True))
        print(f"\n--- Found {len(elements)} raw elements ---")

        table_elements = [el for el in elements if isinstance(el, Table)]
        text_elements = [el for el in elements if not isinstance(el, Table)]

        print(f"--- Separated content into {len(table_elements)} tables and {len(text_elements)} text elements. ---")
        return table_elements, text_elements

    def _process_table_elements(self, table_elements: List, pdf_file_path: pathlib.Path, company_ticker: str) -> List[
        Document]:
        """
        Processes a list of table elements into Document objects.
        """
        table_chunks = []
        for table_el in table_elements:
            table_text = getattr(table_el.metadata, 'text_as_html', None) or table_el.text
            cleaned_text = _clean_element_text(table_text)  # Assumes _clean_element_text exists

            context = Context(
                pdf_title=pdf_file_path.stem,
                page_number=1,
                section_title=getSectionTitle(1),
                relevant_keywords=getRelevantKeywords()
            )

            new_chunk = Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_file_path.name,
                    "page": table_el.metadata.page_number,
                    "company": company_ticker,
                    "element_type": "table"
                }
            )
            table_chunks.append(new_chunk)
        return table_chunks

    def _process_text_elements(self, text_elements: List, pdf_file_path: pathlib.Path, company_ticker: str) -> List[
        Document]:
        """
        Chunks a list of text elements and processes them into Document objects.
        """
        text_chunks_raw = chunk_by_title(
            text_elements,
            max_characters=1000,
            new_after_n_chars=800,
            combine_text_under_n_chars=500,
            overlap=200,
            overlap_all=True
        )

        final_text_chunks = []
        for chunk in text_chunks_raw:
            cleaned_text = _clean_element_text(chunk.text)  # Assumes _clean_element_text exists

            # Filter out very short, likely irrelevant, text snippets
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
                final_text_chunks.append(new_chunk)
        return final_text_chunks

    # !!! Main Function !!!
    def _load_pdf(self, pdf_file_path: pathlib.Path) -> List[Document]:
        """
        Loads, partitions, cleans, and chunks a PDF file into a list of Documents.
        """
        table_elements, text_elements = self._partition_and_separate_elements(pdf_file_path)

        if not table_elements and not text_elements:
            return []

        company_ticker = pdf_file_path.stem.split('-')[0].upper()

        table_chunks = self._process_table_elements(table_elements, pdf_file_path, company_ticker)
        text_chunks = self._process_text_elements(text_elements, pdf_file_path, company_ticker)

        final_chunks = table_chunks + text_chunks

        print(f"--- Finished processing. Generated {len(final_chunks)} total chunks for {pdf_file_path.name}. ---\n")

        return final_chunks

    def load(self) -> List[Document]:
        return self._load_pdf(self.file_path)