from langchain_openai import ChatOpenAI

from finquery_app.config import (
    LMSTUDIO_SMART_MODEL_NAME, LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    COLLECTION_NAME, CHROMA_DB_PATH
)
from finquery_app.chains.conversational_refiner import create_conversational_refiner_chain
from finquery_app.chains.answer_chain import create_rag_chain  # Import the fixed chain
from finquery_app.manager import get_vector_store, get_embeddings
from finquery_parser.types import PostgresDBConnector


def run_interactive_session():
    """
    Initializes all components and runs a full, end-to-end interactive
    chat session from query refinement to final answer retrieval.
    """
    print("--- Initializing Interactive RAG Session ---")

    try:
        llm = ChatOpenAI(
            temperature=0.1,
            model=LMSTUDIO_SMART_MODEL_NAME,
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY
        )
        db_connector = PostgresDBConnector(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        embeddings = get_embeddings()
        vector_store = get_vector_store(COLLECTION_NAME, embeddings, CHROMA_DB_PATH)
        print("Components initialized successfully.")
    except Exception as e:
        print(f"!!! CRITICAL ERROR during initialization: {e}")
        return

    refiner_chain = create_conversational_refiner_chain(llm, db_connector)

    session_id = "interactive_runner_01"
    print(f"\n--- Starting Chat Session (ID: {session_id}) ---")
    print("Type 'exit' or 'quit' to end the session.")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            response = refiner_chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            action = response.get("action")

            if action == "ask":
                print(f"Assistant: {response.get('question', 'Could you clarify?')}")

            elif action == "filter":
                print("Assistant: I have enough information. Searching for the answer...")

                query_data = response.get("data", {})
                search_query = query_data.get("search_query", user_input)
                metadata_filter = query_data.get("metadata_filter", {})

                # This now calls the fixed RAG chain creator
                rag_chain = create_rag_chain(vector_store, llm, metadata_filter)

                print("\n--- Final Answer ---")
                final_answer = rag_chain.invoke(search_query)
                print(final_answer)
                print("--------------------\n")

            else:
                print(f"Assistant (Error): Received an unknown or malformed response:\n{response}")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during the chat session: {e} ---")
    finally:
        if db_connector:
            db_connector.close()
        print("--- Chat Session Finished ---")


if __name__ == "__main__":
    run_interactive_session()
