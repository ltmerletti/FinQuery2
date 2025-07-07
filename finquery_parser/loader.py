import pathlib
from typing import List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .utils import load_pdf


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def load(self) -> List[Document]:
        return load_pdf(self.file_path)
