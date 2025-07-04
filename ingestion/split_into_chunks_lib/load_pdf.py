import pathlib
from typing import List

from langchain_core.documents import Document

from ingestion.split_into_chunks_lib.Context import Context
from ingestion.split_into_chunks_lib.get_relevant_keywords import get_relevant_keywords
from ingestion.split_into_chunks_lib.partition_and_separate_elements import partition_and_separate_elements

import htmltabletomd

import html

from unstructured.chunking.title import chunk_by_title

def load_pdf(pdf_file_path: pathlib.Path) -> List[Document]:
    (table_elements, table_contextss), (text_elements, text_contextss) = partition_and_separate_elements(
        pdf_file_path)

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

        new_doc = Document(
            page_content=augmented_content,
            metadata={
                "source": pdf_file_path.name,
                "page": ccontext.page_number,
                "company": company_ticker,
                "element_type": "Table"
            }
        )
        final_chunkss.append(new_doc)

    print(f"Recursively chunking and augmenting {len(text_elements)} text elements...")
    if text_elements:
        text_chunks_from_title = chunk_by_title(
            text_elements,
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=400,
        )

        for chunkk in text_chunks_from_title:
            chunk_context = Context(
                pdf_title=pdf_file_path.stem,
                page_number=getattr(chunkk.metadata, 'page_number', None),
                section_title=getattr(chunkk.metadata, 'filename', 'Misc. Text'),
                element_type="Text"
            )

            chunk_context.relevant_keywords = get_relevant_keywords(chunkk, chunk_context)

            context_string = chunk_context.to_string()
            content_string = chunkk.text
            augmented_content = f"{context_string}\n\n[CONTENT]\n{content_string}"

            new_doc = Document(
                page_content=augmented_content,
                metadata={
                    "source": pdf_file_path.name,
                    "page": chunk_context.page_number,
                    "company": company_ticker,
                    "element_type": "Text"
                }
            )
            final_chunkss.append(new_doc)

    print(f"--- Finished processing. Generated {len(final_chunkss)} total chunks. ---")

    return final_chunkss
