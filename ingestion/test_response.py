from pprint import pprint

from langchain_chroma import Chroma
from chromainit.database_setup import get_embeddings


def test_simple_query(query_text: str, n_results: int = 5):
    if not query_text:
        print("Query text cannot be empty.")
        return None

    vector_store = Chroma(
        collection_name="financial_documents",
        embedding_function=get_embeddings(),
        persist_directory="../chromadb"
    )

    target_company = "AAPL"
    metadata_filter = {"company": target_company}

    retriever = vector_store.as_retriever(
        search_kwargs={'filter': metadata_filter}
    )

    results = retriever.invoke(query_text)

    return results


if __name__ == "__main__":
    rag_test_questions = [
        # Direct Data Retrieval
        "What were Apple's total net sales in 2023?",
        "How much did Apple spend on Research and Development in 2023?",
        "What was the net income for fiscal year 2022?",
        "Find the total assets listed on the Consolidated Balance Sheets for 2023.",
        "What were the net sales for the Mac product line in 2023?",

        # Comparative & Trend Analysis
        "How did iPhone net sales in 2023 compare to 2022?",
        "Did Services net sales increase or decrease from 2022 to 2023? By how much?",
        "What was the percentage change in Mac net sales between 2023 and 2022?",
        "Compare the net sales of the Americas region versus the Europe region in 2023.",
        "What was the trend for Wearables, Home and Accessories sales over the last three fiscal years shown?",

        # Narrative & Explanatory Retrieval
        "According to the document, why did Mac net sales decrease in 2023?",
        "What factors contributed to the change in total net sales in 2023 compared to the previous year?",
        "What were some of the significant product announcements in the first quarter of fiscal year 2023?",
        "What does the document say about the impact of foreign currency weakness on net sales?",
        "Describe the company's historical seasonality in sales.",

        # Deeper & Component-Based Questions
        "What were the net sales for the 'Rest of Asia Pacific' region in 2023?",
        "Break down the Services net sales for 2023 if possible.",
        "What are the main components of 'Cost of Sales'?",
        "What were the total liabilities and shareholders' equity for 2023?",
        "How much cash and cash equivalents did Apple have at the end of the 2023 fiscal year?",
    ]

    for question in rag_test_questions:
        pprint(test_simple_query(question))
