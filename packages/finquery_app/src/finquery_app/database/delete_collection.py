import chromadb
import pathlib
import argparse

from finquery_app.config import CHROMA_DB_PATH


def delete_collection_and_folder(collection_name: str, persist_directory: str):
    """Deletes a ChromaDB collection and its corresponding data folder."""
    print(f"--- Attempting to delete collection '{collection_name}' ---")
    persist_path = pathlib.Path(persist_directory)

    if not persist_path.exists() or not persist_path.is_dir():
        print(f"Error: Persistence directory not found at '{persist_directory}'.")
        return

    try:
        client = chromadb.PersistentClient(path=str(persist_path))
        print(f"Successfully connected to client at '{persist_directory}'.")

        client.delete_collection(name=collection_name)
        print(f"Successfully deleted collection '{collection_name}' from ChromaDB client.")

    except ValueError:
        print(f"Collection '{collection_name}' does not exist. No action taken.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Delete a ChromaDB collection and its data.")
    parser.add_argument("collection_name", type=str, help="The name of the collection to delete.")
    args = parser.parse_args()

    print(f"This script will permanently delete the collection '{args.collection_name}'")
    print(f"from the directory: '{CHROMA_DB_PATH}'")

    confirm = input("Are you sure you want to proceed? (y/n): ")
    if confirm.lower() == 'y':
        delete_collection_and_folder(
            collection_name=args.collection_name,
            persist_directory=str(CHROMA_DB_PATH)
        )
        print("\n--- Deletion process complete. ---")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()