import pathlib

from langchain_openai import ChatOpenAI
from spacy import load
from tiktoken import get_encoding

from finquery_app.config import LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, LMSTUDIO_MODEL_NAME
from finquery_parser.utils import load_pdf


def main():
    """
    Main function to execute the PDF loading and processing pipeline.
    """
    pdf_to_process = pathlib.Path("../../../../../../../reports/pltr-20231231.pdf")

    llm_base_url = LMSTUDIO_BASE_URL
    llm_api_key = LMSTUDIO_API_KEY
    llm_model_name = LMSTUDIO_MODEL_NAME

    print(f"Initializing LLM with base URL: {llm_base_url} and model: {llm_model_name}")

    llm = ChatOpenAI(model=llm_model_name, api_key=llm_api_key, base_url=llm_base_url, temperature=0.1)

    print(f"Starting the loading process for: {pdf_to_process.name}\n")

    nlp, tiktoken_encoding = None, None

    try:
        nlp = load("en_core_web_sm")
        tiktoken_encoding = get_encoding("cl100k_base")
    except Exception as e:
        print(f"ERROR loading tiktoken model: {e}")

    documents = load_pdf(pdf_to_process, llm=llm, use_high_res=False, nlp=nlp, tokenizer=tiktoken_encoding)

    if not documents:
        print("\nNo documents were processed or returned from the loader.")
        return

    print(f"\n✅ Successfully processed and chunked the document into {len(documents)} chunks.")
    print("=======================================================================")
    print("                          Processed Chunks")
    print("=======================================================================\n")

    for i, doc in enumerate(documents):
        print(f"--- Chunk {i + 1}/{len(documents)} ---")
        print(f"Type:     {doc.metadata.get('element_type', 'N/A')}")
        print(f"Source:   {doc.metadata.get('source', 'N/A')}")
        print(f"Section:  {doc.metadata.get('section', 'N/A')}")
        print(f"Keywords: {doc.metadata.get('keywords', 'N/A')}")
        print("\n--- Content ---\n")
        print(doc.page_content)
        print("\n-----------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
