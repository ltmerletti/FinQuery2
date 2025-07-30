import json
import re
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory

from finquery_app.config import CONVERSATIONAL_QUERY_REFINER_PROMPT
from finquery_parser.types import PostgresDBConnector


def strip_thinking_tags(text: str) -> str:
    """Removes <think>...</think> tags from an LLM response string."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _get_combined_context(db_connector: PostgresDBConnector) -> dict:
    """
    Private helper function to fetch and combine different pieces of
    database context into a single dictionary for the prompt.
    """
    # Fetch the two separate pieces of context
    doc_types = db_connector.get_known_doc_types()
    recent_docs = db_connector.get_recent_documents_summary()  # We will add this simple method next

    # Combine them into the format the prompt expects
    return {
        "document_types": doc_types,
        "recent_documents": recent_docs
    }


def create_conversational_refiner_chain(llm: ChatOpenAI, db_connector: PostgresDBConnector):
    """
    Creates a stateful, conversational chain that refines a user's query
    using a rich, two-part, real-time database context.
    """
    prompt = ChatPromptTemplate.from_template(CONVERSATIONAL_QUERY_REFINER_PROMPT)
    parser = JsonOutputParser()

    base_chain = prompt | llm | RunnableLambda(lambda x: strip_thinking_tags(x.content)) | parser

    store = {}  # Replace with persistent storage in production

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def run_chain_with_dynamic_context(inputs: dict):
        """Wrapper to inject the two-part database context."""
        # Use our new helper to get the combined context
        query_context = _get_combined_context(db_connector)

        inputs_with_context = {
            **inputs,
            "document_types": json.dumps(query_context.get("document_types", []), indent=2),
            "recent_documents": json.dumps(query_context.get("recent_documents", []), indent=2),
        }

        return base_chain.invoke(inputs_with_context)

    context_aware_chain = RunnableLambda(run_chain_with_dynamic_context)

    conversational_chain = RunnableWithMessageHistory(
        context_aware_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return conversational_chain
