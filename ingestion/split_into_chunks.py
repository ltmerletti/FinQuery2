import pathlib
import re
import uuid

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from ingestion.create_multi_representation import get_MRI, TableRepresentations
from ingestion.heuristic import isHighValue


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


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def _load_pdf(self, pdf_file_path: pathlib.Path) -> tuple[list[Document], list[Document]]:
        print(f"Partitioning document: {pdf_file_path}")

        if not str(pdf_file_path).endswith(".pdf"):
            return [], []

        elements = list(partition_pdf(pdf_file_path, strategy="hi_res", infer_table_structure=True))
        print(f"\n--- Found {len(elements)} Elements ---")

        try:
            company_ticker = pdf_file_path.stem.split('-')[0].upper()
            if len(company_ticker) <= 2 or len(company_ticker) >= 7:
                print(f"Filename '{pdf_file_path.name}' may not be a valid ticker. Discarding.")
                company_ticker = ""
        except Exception:
            print(f"Error finding company ticker for {pdf_file_path.name}. Is the file named correctly?")
            company_ticker = ""

        table_elements = [el for el in elements if isinstance(el, Table)]
        text_elements = [el for el in elements if not isinstance(el, Table)]

        print(f"!!! Separated content into {len(table_elements)} tables and {len(text_elements)} text elements. !!!")

        child_docs = []
        representation_docs = []

        for table_el in table_elements:
            table_text = getattr(table_el.metadata, 'text_as_html', table_el.text) or table_el.text

            if isHighValue(table_el, table_text):
                try:
                    multi_rep: TableRepresentations = get_MRI(table_text)
                except Exception as e:
                    print(
                        f"ERROR: LLM call failed for a high-value table on page {table_el.metadata.page_number}. Skipping MRI. Error: {e}")
                    representation_docs.append(Document(page_content=_clean_element_text(table_text),
                                                        metadata={"source": pdf_file_path.name,
                                                                  "page": table_el.metadata.page_number,
                                                                  "company": company_ticker,
                                                                  "element_type": "table_fallback"}))
                    continue

                doc_id = str(uuid.uuid4())
                child_doc = Document(page_content=table_text,
                                     metadata={"source": pdf_file_path.name, "page": table_el.metadata.page_number,
                                               "company": company_ticker, "element_type": "table_original",
                                               "doc_id": doc_id})
                child_docs.append(child_doc)

                summary_doc = Document(page_content=multi_rep.summary,
                                       metadata={"source": pdf_file_path.name, "page": table_el.metadata.page_number,
                                                 "company": company_ticker, "element_type": "table_summary",
                                                 "doc_id": doc_id})
                representation_docs.append(summary_doc)

                for q in multi_rep.hypothetical_questions:
                    question_doc = Document(page_content=q, metadata={"source": pdf_file_path.name,
                                                                      "page": table_el.metadata.page_number,
                                                                      "company": company_ticker,
                                                                      "element_type": "table_question",
                                                                      "doc_id": doc_id})
                    representation_docs.append(question_doc)
            else:
                plain_table_doc = Document(page_content=_clean_element_text(table_text),
                                           metadata={"source": pdf_file_path.name,
                                                     "page": table_el.metadata.page_number, "company": company_ticker,
                                                     "element_type": "table"})
                representation_docs.append(plain_table_doc)

        text_chunks = chunk_by_title(text_elements, max_characters=1000, new_after_n_chars=800,
                                     combine_text_under_n_chars=500, overlap=200, overlap_all=True)

        for chunk in text_chunks:
            cleaned_text = _clean_element_text(chunk.text)
            if len(cleaned_text) > 50:
                new_doc = Document(page_content=cleaned_text,
                                   metadata={"source": pdf_file_path.name, "page": chunk.metadata.page_number,
                                             "company": company_ticker, "element_type": "text"})
                representation_docs.append(new_doc)

        print(
            f"--- Finished processing. Generated {len(child_docs)} original tables and {len(representation_docs)} text representations. ---\n")

        return child_docs, representation_docs

    def load(self) -> tuple[list[Document], list[Document]]:
        return self._load_pdf(self.file_path)
