import pathlib
import re
from typing import List
from chromainit.chroma_collection_document import DocumentChunk
from unstructured.partition.pdf import partition_pdf

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table


# regex function to clean element text
def clean_element_text(text: str) -> str:
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


def load_pdf(pdf_file_path: pathlib.Path) -> List[DocumentChunk]:
    print(f"Partitioning document: {pdf_file_path}")

    if not str(pdf_file_path).endswith(".pdf"):
        return []

    elements = list(partition_pdf(pdf_file_path, strategy="hi_res", infer_table_structure=True))

    print("\n--- Found Elements ---")

    final_chunks = []

    pdf_file_path.stem.split('-')[0].upper()

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

        cleaned_text = clean_element_text(table_text)

        new_chunk = DocumentChunk(
            text=cleaned_text,
            source_document=pdf_file_path.name,
            page_number=table_el.metadata.page_number
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
        cleaned_text = clean_element_text(chunk.text)

        if len(cleaned_text) > 50:
            new_chunk = DocumentChunk(
                text=cleaned_text,
                source_document=pdf_file_path.name,
                page_number=chunk.metadata.page_number
            )
            final_chunks.append(new_chunk)

    print(f"--- Finished processing. Generated {len(final_chunks)} total chunks for {pdf_file_path.name}. ---\n")

    return final_chunks
