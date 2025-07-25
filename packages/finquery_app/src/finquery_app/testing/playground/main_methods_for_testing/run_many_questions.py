import time

from langchain_core.runnables import RunnableConfig

from finquery_app.chains.answer_chain import create_rag_chain
from finquery_app.config import CHROMA_DB_PATH, COLLECTION_NAME, LMSTUDIO_SMART_MODEL_NAME
from finquery_app.manager import get_langfuse_callback, get_llm
from finquery_app.manager import get_vector_store, get_embeddings
from finquery_app.querying.query import get_rag_test_questions


def main():
    print("--- Initializing FinQuery Components ---")

    embeddings = get_embeddings()
    vector_store = get_vector_store(COLLECTION_NAME, embeddings, str(CHROMA_DB_PATH))
    langfuse_handler = get_langfuse_callback()
    llm = get_llm(model_name=LMSTUDIO_SMART_MODEL_NAME)

    config = RunnableConfig(callbacks=[langfuse_handler], run_name="testing_chain_stuff")

    print("\n--- 🔗 Creating RAG Chain (once) ---")
    try:
        rag_chain = create_rag_chain(vector_store, llm)
        print("RAG Chain created successfully.")
    except Exception as e:
        print(f"Failed to create RAG chain: {e}")
        return

    questions = ["What is the change in foreign currency translation, net of tax for Apple in September 30, 2023"]
    total_questions = len(questions)

    print(f"\n--- Processing {total_questions} questions ---")

    for i, question in enumerate(questions):
        print(f"\n[{i + 1}/{total_questions}] QUERY: {question}")
        try:
            answer = rag_chain.invoke(question, config=config)
            print(f"ANSWER: {answer}")
            print(f"(Took ? seconds)")

        except Exception as e:
            print(f"An error occurred while processing the question: {e}")
            continue

        if (i + 1) % 10 == 0 and (i + 1) < total_questions:
            print("\n--- Pausing for 1 minutes (60 seconds) to let computer cool down... ---")
            time.sleep(60)
            print("--- ▶️ Resuming processing. ---")

    print(f"\n--- Finished processing all questions in ? seconds. ---")


if __name__ == '__main__':
    main()
