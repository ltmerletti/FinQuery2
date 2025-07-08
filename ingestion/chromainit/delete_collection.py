import chromadb
import pathlib
import shutil
import sys

project_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def delete_collection_and_folder(collection_name: str, persist_directory: str):
    print(f"--- Attempting to delete collection '{collection_name}' ---")
    persist_path = pathlib.Path(persist_directory)

    if not persist_path.exists() or not persist_path.is_dir():
        print(f"Error: Persistence directory not found at '{persist_directory}'.")
        return

    try:
        client = chromadb.PersistentClient(path=str(persist_path))
        print(f"Successfully connected to client at '{persist_directory}'.")

        try:
            collection = client.get_collection(name=collection_name)
            collection_id = collection.id
            print(f"Found collection '{collection_name}' with ID: {collection_id}")

            client.delete_collection(name=collection_name)
            print(f"Successfully deleted collection '{collection_name}' from ChromaDB client.")

            collection_folder_path = persist_path / str(collection_id)
            if collection_folder_path.exists() and collection_folder_path.is_dir():
                shutil.rmtree(collection_folder_path)
                print(f"Successfully removed data folder: '{collection_folder_path}'")
            else:
                print(
                    f"Warning: Data folder for collection ID '{collection_id}' not found. It might have been deleted previously.")

        except ValueError:
            print(f"Collection '{collection_name}' does not exist in the database. No action taken.")
            collections = client.list_collections()
            if collections:
                print(f"Available collections: {[c.name for c in collections]}")
            else:
                print("No collections found in the database.")


    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    db_directory = project_root / "chromadb"
    collection_to_delete = "financial_documents"

    print(f"This script will permanently delete the collection '{collection_to_delete}'")
    print(f"and its data from the directory: '{db_directory}'")

    confirm = input("Are you sure you want to proceed? (y/n): ")
    if confirm.lower() == 'y':
        delete_collection_and_folder(
            collection_name=collection_to_delete,
            persist_directory=str(db_directory)
        )
        print("\n--- Deletion process complete. ---")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()
