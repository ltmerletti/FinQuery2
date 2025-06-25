import pathlib
from typing import List
from chromainit.chroma_collection_document import DocumentChunk
from unstructured.partition.pdf import partition_pdf

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table, CompositeElement


# TODO: change chunking to ensure that it works by paragraph, and that a table is never > 1 chunk

def load_pdf(pdf_file_path: pathlib.Path) -> List[DocumentChunk]:
    print(f"Partitioning document: {pdf_file_path}")

    if not str(pdf_file_path).endswith(".pdf"):
        return []

    elements = list(partition_pdf(pdf_file_path, strategy="hi_res"))

    print("\n--- Found Elements ---")

    chunks_from_unstructured = chunk_by_title(
        elements,
        max_characters=1000,
        new_after_n_chars=800,
        combine_text_under_n_chars=500,
    )

    final_chunks = []
    for chunk in chunks_from_unstructured:
        if isinstance(chunk, Table):
            print("--- Found a Table Chunk ---")

        if isinstance(chunk, CompositeElement):
            print(f"--- Found a Text Chunk (Size: {len(chunk.text)}) ---")

        new_chunk = DocumentChunk(
            text=chunk.text,
            source_document=chunk.metadata.filename,
            page_number=chunk.metadata.page_number
        )
        final_chunks.append(new_chunk)

    return final_chunks
