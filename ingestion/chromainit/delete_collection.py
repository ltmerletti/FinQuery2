import chromadb
import argparse
import pathlib


def delete_chroma_collection(collection_name: str, db_path: str):
    try:
        client = chromadb.PersistentClient(path=db_path)

        print(f"Attempting to delete collection: '{collection_name}' from database at '{db_path}'...")

        client.delete_collection(name=collection_name)

        print(f"Collection '{collection_name}' deleted successfully.")

    except ValueError as e:
        print(f"Collection '{collection_name}' may not have existed, which is okay.")
        print(f"Underlying message: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a ChromaDB collection.")
    parser.add_argument(
        "--name",
        type=str,
        default="financial_documents",
        help="The name of the collection to delete."
    )
    args = parser.parse_args()

    project_root = pathlib.Path(__file__).parent.parent.parent
    db_directory = str(project_root / "chromadb")

    delete_chroma_collection(args.name, db_directory)
