from typing import List
from chromainit.chroma_collection_document import DocumentChunk

from unstructured.partition.auto import partition


def load_pdf(pdf_file_path) -> List[DocumentChunk]:
    print(f"Partitioning document: {pdf_file_path}")

    elements = list(partition(pdf_file_path, strategy="hi_res"))

    print("\n--- Found Elements ---")

    print("\n\n".join([str(el) for el in elements[:3]]))

    if len(elements) > 3:
        omitted_count = len(elements) - 3
        print(f"\n\n... ({omitted_count} more elements omitted for brevity) ...")
    chunks = []

    for el in elements:
        chunk = DocumentChunk(
            text=el.text,
            source_document=el.metadata.filename,
            page_number=el.metadata.page_number
        )
        chunks.append(chunk)

    return chunks
