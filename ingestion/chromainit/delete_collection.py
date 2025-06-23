import chromadb


def delete_chroma_collection(collection_name: str, db_path: str = "./chromadb"):
    try:
        client = chromadb.PersistentClient(path=db_path)

        print(f"Attempting to delete collection: '{collection_name}'...")

        client.delete_collection(name=collection_name)

        print(f"Collection '{collection_name}' deleted successfully.")

    except ValueError as e:
        print(f"Collection '{collection_name}' may not have existed, which is okay.")
        print(f"Underlying message: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    collection_to_delete = "d27b771f-aba3-41ed-a4a3-b06a9b891589"

    delete_chroma_collection(collection_to_delete)
