import pathlib
import shutil
from typing import List

from langchain.indexes import index
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.indexes import SQLRecordManager

from finquery_app.ingestion.find_file_paths import get_file_paths

from finquery_parser.loader import CustomPDFLoader

def run_ingestion_process(vector_store: Chroma, record_manager: SQLRecordManager):
    """
    Finds, processes, and indexes all new PDF documents from the 'reports'
    directory into the vector store. This is the core, reusable logic.
    """
    # Establish paths relative to this file's location to find the project root
    project_root = pathlib.Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"

    file_paths = get_file_paths(str(reports_dir))
    if not file_paths:
        print("No new PDF files found in the 'reports' directory. Exiting.")
        return

    print("Files to be processed:")
    for file in file_paths:
        print(f" - {file.name}")

    for file_path in file_paths:
        print(f"\n========================================")
        print(f"STARTING PROCESSING FOR: {file_path.name}")
        print(f"========================================")

        try:
            loader: CustomPDFLoader = CustomPDFLoader(str(file_path))
            docs: List[Document] = loader.load()

            if not docs:
                print(f"No documents were generated for {file_path.name}. Skipping.")
                continue

            print(f"Indexing {len(docs)} chunks for {file_path.name}...")
            index(
                docs,
                record_manager,
                vector_store,
                cleanup="incremental",
                source_id_key="source",
                batch_size=64
            )
            print(f"Successfully indexed {file_path.name}.")

            # Move the processed file to the 'added' subdirectory
            destination_dir = reports_dir / 'added'
            destination_dir.mkdir(exist_ok=True)
            shutil.move(file_path, destination_dir / file_path.name)
            print(f"File {file_path.name} has been moved to '{destination_dir.name}' directory.")

        except Exception as e:
            print(f"!!! An error occurred while processing {file_path.name}: {e}")
            continue

    print("\n--- Finished processing all files ---")
    final_count = vector_store._collection.count()
    print(f"The collection now has {final_count} total chunks.")
