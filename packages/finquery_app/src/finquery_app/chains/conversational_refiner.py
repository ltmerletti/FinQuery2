import json
import re
from typing import Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Assuming the prompt is defined in a central config or prompts file
from finquery_app.config import CONVERSATIONAL_QUERY_REFINER_PROMPT
from finquery_parser.types import PostgresDBConnector


def strip_thinking_tags(text: str) -> str:
    """Removes <think>...</think> tags from an LLM response string."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _format_history_for_prompt(history: ChatMessageHistory) -> str:
    """Formats the chat history into a simple string for the prompt."""
    if not history.messages:
        return "No history yet."

    formatted_messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_messages.append(f"AI: {msg.content}")
    return "\n".join(formatted_messages)


class ConversationalRefiner:
    """
    A stateful class that manages the conversational query refinement process.
    Using a class ensures that the chat history store (`self.store`) persists
    reliably across multiple calls within the same session.
    """

    def __init__(self, llm: ChatOpenAI, db_connector: PostgresDBConnector):
        self.llm = llm
        self.db_connector = db_connector
        self.prompt = ChatPromptTemplate.from_template(CONVERSATIONAL_QUERY_REFINER_PROMPT)
        self.parser = JsonOutputParser()
        self.base_chain = self.prompt | self.llm | RunnableLambda(lambda x: strip_thinking_tags(x.content))
        self.store: Dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Retrieves a chat session history, creating one if it doesn't exist."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _get_combined_context(self) -> Dict[str, Any]:
        """Fetches and combines database context for the prompt."""
        doc_types = self.db_connector.get_known_doc_types()
        recent_docs = self.db_connector.get_recent_documents_summary()
        return {"document_types": doc_types, "recent_documents": recent_docs}

    def invoke(self, inputs: dict, config: dict):
        """
        Manages a single turn of the conversation: fetches context and history,
        invokes the LLM, and updates the history.
        """
        session_id = config["configurable"]["session_id"]

        history = self.get_session_history(session_id)
        query_context = self._get_combined_context()

        inputs_with_context = {
            "question": inputs["question"],
            "chat_history": _format_history_for_prompt(history),
            "document_types": json.dumps(query_context.get("document_types", []), indent=2),
            "recent_documents": json.dumps(query_context.get("recent_documents", []), indent=2),
        }

        ai_response_str = self.base_chain.invoke(inputs_with_context)

        history.add_user_message(inputs["question"])
        history.add_ai_message(ai_response_str)

        return self.parser.invoke(ai_response_str)


def create_conversational_refiner_chain(llm: ChatOpenAI, db_connector: PostgresDBConnector):
    """
    Factory function that creates an instance of the ConversationalRefiner
    and returns its invoke method as a runnable.
    """
    refiner = ConversationalRefiner(llm, db_connector)
    return RunnableLambda(refiner.invoke)
