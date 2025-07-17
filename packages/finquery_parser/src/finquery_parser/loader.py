import pathlib
from typing import List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from spacy.language import Language
from tiktoken.core import Encoding


from .utils import load_pdf


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str, llm: ChatOpenAI, nlp: Language, tokenizer: Encoding):
        self.file_path = pathlib.Path(file_path)
        self.llm = llm
        self.nlp = nlp
        self.tokenizer = tokenizer

    def load(self) -> List[Document]:
        return load_pdf(self.file_path, nlp=self.nlp, tokenizer=self.tokenizer, llm=self.llm)
