import pathlib
from typing import List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from ingestion.split_into_chunks_lib.load_pdf import load_pdf


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def load(self) -> List[Document]:
        return load_pdf(self.file_path)


if __name__ == "__main__":
    test_file_path = "/Users/lukem/PycharmProjects/FinQuery2/reports/aapl-20230930.pdf"

    print(f"--- INITIALIZING LOADER FOR: {test_file_path} ---")

    loader = CustomPDFLoader(file_path=test_file_path)

    final_chunks: List[Document] = loader.load()
    for chunk in final_chunks:
        print(chunk)
        print("---")
