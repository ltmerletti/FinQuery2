from typing import List, Dict, Any

# Langchain and Third-Party Imports
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

# Local Application Imports
from finquery_app.config import RERANKING_MODEL_NAME, FIND_ANSWER_FROM_RETRIEVALS_TEMPLATE

# --- RAG Chain Components ---

class QwenReranker(BaseCrossEncoder):
    """A wrapper for the Sentence Transformers Cross-Encoder model for re-ranking."""
    def __init__(self, model_name=RERANKING_MODEL_NAME, device="cpu"):
        print(f"Loading the Qwen re-ranking model '{model_name}' into memory...")
        self.model = CrossEncoder(model_name, device=device)
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.model.config.pad_token_id = self.model.tokenizer.eos_token_id
        print("Re-ranking model loaded and configured.")

    def score(self, text_pairs: List[List[str]]) -> List[float]:
        """Calculates the relevance scores for text pairs."""
        return self.model.predict(text_pairs).tolist()

def format_docs_with_metadata(docs: List[Document]) -> str:
    """Formats retrieved documents into a string for the final prompt."""
    if not docs:
        return "No relevant documents were found to answer the question."
    formatted_snippets = []
    for doc in docs:
        try:
            content_start_index = doc.page_content.index("[CONTENT]")
            content = doc.page_content[content_start_index:].replace("[CONTENT]", "").strip()
        except ValueError:
            content = doc.page_content.strip()
        source = doc.metadata.get("source", "Unknown Source")
        snippet = f"Source: {source}\nContent:\n{content}"
        formatted_snippets.append(snippet)
    return "\n\n---\n\n".join(formatted_snippets)

def _format_filter_for_chroma(metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a simple key-value filter into the format ChromaDB expects for multi-condition queries."""
    if not metadata_filter:
        return {}
    if len(metadata_filter) == 1:
        return metadata_filter
    return {"$and": [{key: value} for key, value in metadata_filter.items()]}

def create_rag_chain(vector_store: Chroma, llm: ChatOpenAI, metadata_filter: Dict[str, Any]):
    """
    Creates the full RAG chain, dynamically configured with correctly formatted metadata filters.
    """
    chroma_filter = _format_filter_for_chroma(metadata_filter)
    print(f"\nCreating RAG chain with formatted Chroma filter: {chroma_filter}")

    search_kwargs = {'k': 10, 'filter': chroma_filter} if chroma_filter else {'k': 10}
    base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    reranker_tool = QwenReranker()
    compressor = CrossEncoderReranker(model=reranker_tool, top_n=4)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    prompt = ChatPromptTemplate.from_template(FIND_ANSWER_FROM_RETRIEVALS_TEMPLATE)

    rag_chain = (
        {"context": compression_retriever | format_docs_with_metadata, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
