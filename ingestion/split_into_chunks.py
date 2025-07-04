import pathlib
from typing import List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from ingestion.split_into_chunks_lib.load_pdf import load_pdf


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def load(self) -> List[Document]:
        return load_pdf(self.file_path)


if __name__ == "__main__":
    test_file_path = "/Users/lukem/PycharmProjects/FinQuery2/reports/aapl-20230930.pdf"

    print(f"--- INITIALIZING LOADER FOR: {test_file_path} ---")

    loader = CustomPDFLoader(file_path=test_file_path)

    try:
        final_chunks = loader.load()

        print("\n\n" + "=" * 80)
        print("||" + " VERIFYING FINAL AUGMENTED CHUNKS ".center(76) + "||")
        print("=" * 80)
        print(f"\nGenerated a total of {len(final_chunks)} chunks.")

        for i, chunk in enumerate(final_chunks):
            print(f"\n--- CHUNK {i + 1} ---")

            print("[METADATA]")
            print(chunk.metadata)

            print("\n[AUGMENTED PAGE_CONTENT]")
            print(chunk.page_content)
            print("-" * 80)

    except FileNotFoundError:
        print(f"ERROR: The file '{test_file_path}' was not found. Please check the path.")
    except Exception as e:
        import traceback

        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
