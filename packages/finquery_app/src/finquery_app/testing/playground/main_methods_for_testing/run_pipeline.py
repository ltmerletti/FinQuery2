from pprint import pprint

from finquery_app.config import (
    LMSTUDIO_FAST_LLM_MODEL_NAME, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    COLLECTION_NAME, CHROMA_DB_PATH
)
from finquery_app.ingestion.pipeline import run_ingestion_process
from finquery_app.manager import get_spacy_model, get_tiktoken_model, get_llm, get_vector_store, get_record_manager, \
    get_embeddings
from finquery_parser.types import PostgresDBConnector


def main():
    """
    Main function to initialize all components and run the ingestion pipeline.
    This script acts as the entry point for the application.
    """
    final_documents = []
    print("--- Starting FinQuery PDF Ingestion Pipeline ---")

    db_connector = None
    try:
        embeddings = get_embeddings()
        vector_store = get_vector_store(COLLECTION_NAME, embeddings, CHROMA_DB_PATH)
        record_manager = get_record_manager(COLLECTION_NAME)
        llm = get_llm()
        small_llm = get_llm(model_name=LMSTUDIO_FAST_LLM_MODEL_NAME)
        spacy_model = get_spacy_model()
        tiktoken_model = get_tiktoken_model()
        db_connector = PostgresDBConnector(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )

        final_documents = run_ingestion_process(
            vector_store=vector_store,
            record_manager=record_manager,
            small_llm=small_llm,
            llm=llm,
            nlp=spacy_model,
            tokenizer=tiktoken_model,
            db_connector=db_connector
        )

    except Exception as e:
        print(f"\n--- A CRITICAL error occurred during pipeline execution: {e} ---")
    finally:
        if db_connector:
            db_connector.close()
        print("\n--- FinQuery PDF Ingestion Pipeline Finished ---")


if __name__ == "__main__":
    main()
