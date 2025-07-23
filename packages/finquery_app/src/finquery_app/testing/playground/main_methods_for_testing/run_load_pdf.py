import pathlib
import os
from pprint import pprint

from langchain_openai import ChatOpenAI

from finquery_app.config import (
    SMALL_LLM_NAME, LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, LLM_NAME,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)
from finquery_app.manager import get_spacy_model, get_tiktoken_model
from finquery_parser.types import PostgresDBConnector
from finquery_parser.utils import load_pdf

def main():
    """
    Main function to orchestrate the PDF processing pipeline.
    """
    print("--- Starting FinQuery PDF Processing Pipeline ---")

    try:
        db_connector = PostgresDBConnector(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
    except Exception as e:
        return

    print("\nInitializing models...")
    try:
        small_llm = ChatOpenAI(
            temperature=0.1,
            model=SMALL_LLM_NAME,
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY
        )
        llm = ChatOpenAI(
            temperature=0.1,
            model=LLM_NAME,
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY
        )

        nlp = get_spacy_model()
        tokenizer = get_tiktoken_model()

        print("Models initialized successfully.")

    except Exception as e:
        print(f"Error during model initialization: {e}")
        db_connector.close()
        return

    # --- 3. Document Processing ---
    PDF_INPUT_PATH = pathlib.Path("/Users/lukem/PycharmProjects/FinQuery2/reports/added/tsla-20240930.pdf")
    print(f"\nProcessing document: {PDF_INPUT_PATH.name}")
    if not PDF_INPUT_PATH.exists():
        print(f"ERROR: File not found at '{PDF_INPUT_PATH}'. Please check the path.")
        db_connector.close()
        return

    try:
        final_documents = load_pdf(
            pdf_file_path=PDF_INPUT_PATH,
            small_llm=small_llm,
            llm=llm,
            nlp=nlp,
            tokenizer=tokenizer,
            db_interface=db_connector,
            use_high_res=False,
            filter_small_elements=True
        )

        print("\n--- Pipeline Finished ---")
        if final_documents:
            print(f"Successfully processed and created {len(final_documents)} documents.")
            for document in final_documents:
                pprint(document)
        else:
            print("No documents were generated. Please check the logs for errors.")

    except Exception as e:
        print(f"An unexpected error occurred during the PDF loading process: {e}")
    finally:
        db_connector.close()


if __name__ == "__main__":
    main()
