import sys

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder

from finquery_app.config import PROJECT_ROOT, RERANKING_MODEL_NAME, FIND_ANSWER_FROM_RETRIEVALS_TEMPLATE


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


def create_rag_chain(vector_store, llm):
    # Logic to be implemented:
    # filtered_kwargs: dict = {}

    # base_retriever = vector_store.as_retriever(search_kwargs={'k': 10, 'filter': {
    #     **filtered_kwargs}})

    base_retriever = vector_store.as_retriever(search_kwargs={'k': 10})


    reranker_tool = QwenReranker()

    compressor = CrossEncoderReranker(model=reranker_tool, top_n=4)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    template = FIND_ANSWER_FROM_RETRIEVALS_TEMPLATE

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = ({"context": compression_retriever | format_docs_with_metadata,
                  "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

    return rag_chain
