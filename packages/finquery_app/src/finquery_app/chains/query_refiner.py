import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory

from finquery_app.config import CONVERSATIONAL_QUERY_REFINER_PROMPT
from finquery_parser.types import PostgresDBConnector


def create_conversational_refiner_chain(llm: ChatOpenAI, db_connector: PostgresDBConnector):
    """
    Creates a stateful, conversational chain that refines a user's query.

    This chain fetches available metadata filters from the database ON EVERY TURN
    to provide real-time context to the LLM. This enables it to ask intelligent,
    context-aware clarifying questions before generating a final query filter.

    Args:
        llm: An initialized ChatOpenAI model instance.
        db_connector: An initialized PostgresDBConnector to fetch context.

    Returns:
        A LangChain runnable that manages the conversational refinement process.
    """
    prompt = ChatPromptTemplate.from_template(CONVERSATIONAL_QUERY_REFINER_PROMPT)
    parser = JsonOutputParser()

    base_chain = prompt | llm | parser

    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def run_chain_with_dynamic_context(inputs: dict):
        db_summary = db_connector.get_dynamic_filter_context()
        db_summary_str = json.dumps(db_summary, indent=2)

        inputs_with_context = {
            **inputs,
            "database_filters_summary": db_summary_str
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
