from dataclasses import dataclass


@dataclass
class DocumentChunk:
    text: str
    source_document: str
    page_number: int
