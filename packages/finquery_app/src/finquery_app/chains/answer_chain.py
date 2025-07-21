import sys

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

from finquery_app.config import LMSTUDIO_API_KEY, LMSTUDIO_BASE_URL, PROJECT_ROOT, RERANKING_MODEL_NAME, \
    LMSTUDIO_SMART_MODEL_NAME


class QwenReranker(BaseCrossEncoder):
    def __init__(self, model_name=RERANKING_MODEL_NAME):
        print("Loading the Qwen model into memory...")
        self.model = CrossEncoder(model_name, device="cpu")
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.model.config.pad_token_id = self.model.tokenizer.eos_token_id
        print("Model loaded and configured.")

    def score(self, text_pairs):
        return self.model.predict(text_pairs).tolist()


sys.path.insert(0, str(PROJECT_ROOT))


def format_docs_with_metadata(docs: list) -> str:
    if not docs:
        return "No documents found."

    formatted_snippets = []
    for doc in docs:
        try:
            content_start_index = doc.page_content.index("[CONTENT]")
            content = doc.page_content[content_start_index:].replace("[CONTENT]", "").strip()
        except ValueError:
            content = doc.page_content.strip()

        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")
        snippet = f"Source: {source}, Page: {page}\nContent:\n{content}"
        formatted_snippets.append(snippet)
    return "\n\n---\n\n".join(formatted_snippets)


def create_rag_chain(vector_store):
    base_retriever = vector_store.as_retriever(search_kwargs={'k': 10})

    reranker_tool = QwenReranker()

    compressor = CrossEncoderReranker(model=reranker_tool, top_n=4)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    template = """You are an expert financial analyst AI. Your task is to provide a precise answer to the user's question based *only* on the context provided from financial documents.

Follow these steps rigorously:
1.  Carefully read the user's question to understand exactly what information is being asked for.
2.  Review each of the context snippets below. Each snippet is from a specific source document and page.
3.  Identify the single snippet that most directly and accurately answers the user's question. Ignore snippets that are only tangentially related or do not contain the specific data point requested.
4.  If no snippet contains the answer, respond with: "I could not find the answer in the provided documents."
5.  If a relevant snippet is found, construct your answer by directly extracting the information. State the fact or figure clearly and concisely.
6.  After providing the answer, you MUST cite your source in the format: "(Source: [filename], Page: [page number])".

Do not add any preamble, conversational text, or information that is not from the provided context.

---
CONTEXT SNIPPETS:
{context}
---
USER QUESTION:
{question}
---
PRECISE ANSWER:"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0.1, model=LMSTUDIO_SMART_MODEL_NAME, base_url=LMSTUDIO_BASE_URL,
                     api_key=LMSTUDIO_API_KEY)

    rag_chain = ({"context": compression_retriever | format_docs_with_metadata,
                  "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

    return rag_chain
