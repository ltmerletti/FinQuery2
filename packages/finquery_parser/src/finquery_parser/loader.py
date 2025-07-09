import pathlib
from typing import List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .utils import load_pdf


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str, llm: ChatOpenAI):
        self.file_path = pathlib.Path(file_path)
        self.llm = llm

    def load(self) -> List[Document]:
        return load_pdf(self.file_path, llm=self.llm)
