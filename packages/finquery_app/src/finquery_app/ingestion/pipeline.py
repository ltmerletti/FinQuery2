# Standard Library Imports
import shutil
import pathlib
from typing import List, Dict, Any

# Langchain and Third-Party Imports
from langchain_chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from spacy.language import Language
from tiktoken.core import Encoding

# Local Application Imports
from finquery_app.config import SOURCE_DATA_DIR
from finquery_app.ingestion.find_file_paths import get_file_paths
from finquery_parser.types import DocumentList, PostgresDBConnector
from finquery_parser.utils import load_pdf

def run_ingestion_process(
    vector_store: Chroma,
    record_manager: SQLRecordManager,
    small_llm: ChatOpenAI,
    llm: ChatOpenAI,
    nlp: Language,
    tokenizer: Encoding,
    db_connector: PostgresDBConnector
):
    """
    Finds, processes, indexes, and saves metadata for all new PDF documents
    from the source directory.
    """
    file_paths = get_file_paths(str(SOURCE_DATA_DIR))
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
            docs = load_pdf(
                pdf_file_path=file_path,
                small_llm=small_llm,
                llm=llm,
                nlp=nlp,
                tokenizer=tokenizer,
                db_interface=db_connector,
                use_high_res=False,
                filter_small_elements=True
            )

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

            if hasattr(docs, 'metadata') and docs.metadata:
                print(f"Inserting metadata for {file_path.name} into the database...")
                try:
                    metadata = docs.metadata
                    document_id = db_connector.get_or_create_document_by_filename(file_path.name)

                    if not document_id:
                        print(f"ERROR: Could not retrieve or create a document ID for {file_path.name}.")
                    else:
                        for key, value in metadata.items():
                            if value is not None:
                                db_connector.insert_metadata_value(
                                    document_id=document_id,
                                    meta_key=key,
                                    meta_value=str(value)
                                )
                        print(f"Successfully inserted {len(metadata)} metadata values for document ID {document_id}.")

                except AttributeError as e:
                    print(f"!!! DATABASE CONNECTOR ERROR: A required method is likely missing: {e}")
                except Exception as e:
                    print(f"!!! DATABASE ERROR while inserting metadata for {file_path.name}: {e}")
            else:
                print("No additional metadata was found to insert into the database.")

            destination_dir = SOURCE_DATA_DIR / 'added'
            destination_dir.mkdir(exist_ok=True)
            shutil.move(str(file_path), destination_dir / file_path.name)
            print(f"File {file_path.name} has been moved to the '{destination_dir.name}' directory.")

        except Exception as e:
            print(f"!!! An unhandled error occurred while processing {file_path.name}: {e}")
            continue

    print("\n--- Finished processing all files ---")
    final_count = vector_store._collection.count()
    print(f"The collection now has {final_count} total chunks.")
