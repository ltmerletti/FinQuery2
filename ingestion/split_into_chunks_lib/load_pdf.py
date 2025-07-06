import html
import pathlib
from typing import List

import htmltabletomd
from langchain_core.documents import Document

from ingestion.split_into_chunks_lib.clean_text import clean_element_text
from ingestion.split_into_chunks_lib.partition_and_separate_elements import partition_and_separate_elements


def load_pdf(pdf_file_path: pathlib.Path) -> List[Document]:
    (table_elements, table_contextss), (text_elements, text_contextss) = partition_and_separate_elements(pdf_file_path)

    if not table_elements and not text_elements:
        return []

    company_ticker = pdf_file_path.stem.split('-')[0].upper()
    final_chunkss: List[Document] = []

    print(f"Augmenting and assembling {len(table_elements)} table chunks...")
    for ttable, ccontext in zip(table_elements, table_contextss):
        context_string = ccontext.to_string()

        content_string = htmltabletomd.convert_table(
            html.unescape(getattr(ttable.metadata, 'text_as_html', ttable.text)))

        augmented_content = f"{context_string}\n\n[CONTENT]\n{content_string}"

        new_doc = Document(page_content=augmented_content,
            metadata={"source": pdf_file_path.name, "page": ccontext.page_number, "company": company_ticker,
                "element_type": "Table"})
        final_chunkss.append(new_doc)

    for ttext, ccontext in zip(text_elements, text_contextss):
        context_string = ccontext.to_string()

        content_string = clean_element_text(ttext.text)

        augmented_content = f"{context_string}\n\n[CONTENT]\n{content_string}"

        new_doc = Document(page_content=augmented_content,
            metadata={"source": pdf_file_path.name, "page": ccontext.page_number, "company": company_ticker,
                "element_type": "Text"})
        final_chunkss.append(new_doc)

    return final_chunkss
