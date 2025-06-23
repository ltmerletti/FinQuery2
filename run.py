# query.py

from chromainit import initialize_chroma
import pprint  # pprint is great for printing complex dictionaries cleanly


def run_query(query_text: str):
    """
    Initializes ChromaDB, runs a query, and prints the results.
    """
    # 1. Get the same collection object you used for ingestion
    collection = initialize_chroma()

    print(f"\n🔍 Querying for: '{query_text}'")

    # 2. Run the query
    results = collection.query(
        query_texts=[query_text],  # A list of question(s) to ask
        n_results=5  # The number of most relevant chunks to return
    )

    # 3. Print the results in a readable way
    print("\n✅ Top 5 Results:")
    for i, doc in enumerate(results['documents'][0]):
        print("-" * 50)
        print(f"Result #{i + 1}")

        # Use pprint to nicely print the metadata
        metadata = results['metadatas'][0][i]
        print("Metadata:")
        pprint.pprint(metadata)

        # Print the distance score (lower is more similar)
        distance = results['distances'][0][i]
        print(f"Distance: {distance:.4f}")

        print("\nRetrieved Chunk:")
        print(doc)

    print("-" * 50)


# --- This is where you run the script ---
if __name__ == "__main__":
    # Define the question you want to ask your documents
    my_question = "What are the main risks related to competition?"

    run_query(my_question)
