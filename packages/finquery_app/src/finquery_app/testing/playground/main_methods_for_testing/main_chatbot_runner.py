import json
from langchain_openai import ChatOpenAI

from finquery_app.chains.query_refiner import create_conversational_refiner_chain
from finquery_app.config import (
    LMSTUDIO_MODEL_NAME, LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)
from finquery_parser.types import PostgresDBConnector


def run_chat_session():
    """
    Initializes all components and runs an interactive command-line chat
    session to demonstrate the conversational query refiner.
    """
    print("--- Initializing Conversational Analyst Assistant ---")

    try:
        llm = ChatOpenAI(
            temperature=0.1,
            model=LMSTUDIO_MODEL_NAME,
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY
        )
        db_connector = PostgresDBConnector(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        print("Components initialized successfully.")
    except Exception as e:
        print(f"!!! CRITICAL ERROR during initialization: {e}")
        return

    refiner_chain = create_conversational_refiner_chain(llm, db_connector)

    session_id = "cli_chat_session_01"
    print(f"\n--- Starting Chat Session (ID: {session_id}) ---")
    print("Type 'exit' or 'quit' to end the session.")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Ending chat session.")
                break

            response = refiner_chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            action = response.get("action")

            if action == "ask":
                clarifying_question = response.get("question", "Sorry, I'm not sure how to proceed. Can you rephrase?")
                print(f"Assistant: {clarifying_question}")

            elif action == "filter":
                print("\nAssistant: Great! I have enough information to proceed.")
                print("--- FINAL QUERY PARAMETERS ---")
                print(json.dumps(response.get("data"), indent=2))
                print("------------------------------\n")
                print("Next step: Pass these parameters to the RAG retrieval chain.")
                break

            else:
                print("Assistant: I seem to have run into an issue. Could you please try rephrasing your request?")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during the chat session: {e} ---")
    finally:
        if db_connector:
            db_connector.close()
        print("--- Chat Session Finished ---")


if __name__ == "__main__":
    run_chat_session()
