from langchain_core.runnables import RunnableConfig

from finquery_app.config import CHROMA_DB_PATH, COLLECTION_NAME
from finquery_app.manager import get_vector_store, get_embeddings
from finquery_app.querying.query import get_rag_test_questions, execute_query


def main():
    vectorstore = get_vector_store(COLLECTION_NAME, get_embeddings(), CHROMA_DB_PATH)

    test_questions = get_rag_test_questions()

    num_to_fetch = 4

    config = RunnableConfig()

    print(f"Executing {len(test_questions)} RAG test questions...\n")

    for i, question in enumerate(test_questions, 1):
        try:
            print(f"Q{i}: {question}")
            result = execute_query(question, vectorstore, num_to_fetch, config)
            print("Answer:\n", result, "\n" + "-" * 80 + "\n")
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            print("-" * 80)


if __name__ == "__main__":
    main()
