import pprint
from ingestion.chromainit.chromainit import initialize_chroma


def test_simple_query(query_text: str, n_results: int = 5):
    if not query_text:
        print("Query text cannot be empty.")
        return

    collection = initialize_chroma()

    print(f"\nSearching for: '{query_text}'")

    # query the collection
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    print(f"\nTop {len(results.get('documents', [[]])[0])} Results:")
    if not results.get('documents', [[]])[0]:
        print("No results found.")
        return

    for i, doc_text in enumerate(results['documents'][0]):
        print("-" * 70)
        print(f"Result #{i + 1}")

        metadata = results['metadatas'][0][i]
        print("Source:")
        pprint.pprint(metadata)

        distance = results['distances'][0][i]
        print(f"Similarity Score: {distance:.4f}")

        print("\nRetrieved Text:")
        indented_text = "\n".join(["  " + line for line in doc_text.splitlines()])
        print(indented_text)

    print("-" * 70)


if __name__ == "__main__":
    question = "What were the total net sales or revenue for the first fiscal quarter of 2024?"

    test_simple_query(question)
