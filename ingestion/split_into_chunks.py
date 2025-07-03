import pathlib
import re
from typing import List, Tuple
import html

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Table, Text, Title, Element

import bs4

import htmltabletomd

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from ingestion.split_into_chunks_lib.Context import Context

# TODO:
# - [x] Convert HTML tables to markdown
# - [x] Add the element type as part of context. May help during retrieval.
# - [x] fix the clean_table_text function so it's more specific to tables.
# - [ ] fix the messy section title yes || no issue
# - [x] add support for the keywords and other important context
# - [ ] add support for 1 line summaries of tables.
# - [x] remove empty tables
# - [ ] fix table of contents semantic density being too high? (test + see if is issue first)

# this pattern gets rid of the 'Apple Inc. | 2023 Form 10-K | 21' stuff we don't want
JUNK_FOOTER_PATTERN = re.compile(r'^.*Form 10-K\s*\|\s*\d+\s*$', re.IGNORECASE | re.MULTILINE)
# this pattern helps identify and discard titles that are just form-like checkboxes.
CHECKBOX_PATTERN = re.compile(r'Yes\s*☒\s*No\s*☐', re.IGNORECASE)
# this filters out sec links
SEC_LINK_PATTERN = re.compile(r'https?://www\.sec\.gov/Archives/edgar/data/.*\.htm', re.IGNORECASE)


def _clean_element_text(textt: str) -> str:
    # removes urls (we don't need them for the type of query)
    textt = re.sub(r'https?://\S+', '', textt, flags=re.MULTILINE)
    # removes sec urls
    textt = re.sub(r'\S*www\.sec\.gov\S*', '', textt, flags=re.MULTILINE)
    # remove page number (which is in format (a number)/(a number)
    textt = re.sub(r'\s*\d+/\d+\s*', '', textt)
    # removes dates and times
    textt = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}(,\s*\d{1,2}:\d{2}\s*(AM|PM)?)?', '', textt)
    # removes stray page numbers
    textt = re.sub(r'^\s*\d+\s*$', '', textt, flags=re.MULTILINE)
    # removes empty lines
    textt = re.sub(r'\n\s*\n', '\n', textt)

    return textt.strip()


def _clean_table_text(textt: str) -> str:
    # removes empty lines
    textt = re.sub(r'\n\s*\n', '\n', textt)

    return textt.strip()


def _is_table_functionally_empty(table_htmll: str) -> bool:
    if not table_htmll:
        return True

    soup = bs4.BeautifulSoup(table_htmll, 'html.parser')

    data_cells = soup.find_all('td')

    for cell in data_cells:
        if cell.get_text(strip=True):
            return False

    return True


# These sets are defined once and used by the function.
# They are deliberately kept short for this example; you can expand them.
FINANCIAL_TERMS = {
    'revenue', 'sales', 'income', 'earnings', 'profit', 'margin', 'assets',
    'liabilities', 'equity', 'debt', 'cash flow', 'expenses', 'costs',
    'amortization', 'depreciation', 'investment', 'securities', 'shares'
}
# A more comprehensive set of stop words, including common financial boilerplate
STOP_WORDS = {
    'a', 'an', 'and', 'the', 'is', 'it', 'in', 'on', 'for', 'of', 'as', 'to', 'inc',
    'was', 'were', 'by', 'with', 'or', 'at', 'from', 'that', 'this', 'llc',
    'company', 'inc', 'corp', 'ltd', 'about', 'after', 'all', 'also', 'been',
    'because', 'but', 'can', 'could', 'did', 'do', 'due', 'has', 'had', 'have',
    'how', 'however', 'into', 'its', 'just', 'may', 'most', 'must', 'not',
    'other', 'our', 'out', 'over', 'said', 'should', 'so', 'some', 'such',
    'than', 'then', 'there', 'these', 'they', 'through', 'under', 'upon',
    'use', 'used', 'using', 'various', 'very', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'while', 'who', 'why', 'will', 'would', 'you',
    'your', 'notes', 'note', 'see', 'title', 'part', 'item'
}


def get_relevant_keywords(element: Element, max_keywords: int = 15) -> List[str]:
    if not element or not element.text.strip():
        return []

    candidates = set()

    # --- Heuristic 1: Specialized Logic for Tables ---
    if isinstance(element, Table):
        table_html = getattr(element.metadata, 'text_as_html', '')
        if table_html:
            soup = bs4.BeautifulSoup(table_html, 'html.parser')
            # Extract text from all header cells (<th>)
            for header in soup.find_all('th'):
                candidates.add(header.get_text(strip=True))
            # Extract text from the first cell (<td>) of each row (<tr>)
            for row in soup.find_all('tr'):
                first_cell = row.find('td')
                if first_cell:
                    candidates.add(first_cell.get_text(strip=True))

    # Add capitalized phrases from the element's main text
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', element.text)
    candidates.update(capitalized_phrases)

    # Add all-caps acronyms
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', element.text)
    candidates.update(acronyms)

    final_keywords = []
    seen_keywords = set()

    sorted_candidates = sorted(list(candidates), key=len, reverse=True)

    for keyword in sorted_candidates:
        kw_clean = keyword.strip(" “”)’'.,:()").replace('’', "'")
        kw_lower = kw_clean.lower()

        if (len(kw_clean) > 2 and
                kw_lower not in STOP_WORDS and
                not kw_lower.isdigit()):

            if kw_lower not in seen_keywords:
                final_keywords.append(kw_clean)
                seen_keywords.add(kw_lower)

        if len(final_keywords) >= max_keywords:
            break

    return final_keywords


class CustomPDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)

    def partition_and_separate_elements(self, pdf_file_path: pathlib.Path) -> Tuple[
        Tuple[List[Table], List[Context]], Tuple[List[Text], List[Context]]]:
        """
        Partitions the PDF, finds the section title for each element,
        and separates them into structured tuples of (elements, contexts).
        """
        print(f"Partitioning document: {pdf_file_path}")

        if not str(pdf_file_path).endswith(".pdf"):
            return ([], []), ([], [])

        elements = list(partition_pdf(pdf_file_path, strategy="hi_res", infer_table_structure=True))
        print(f"\n--- Found {len(elements)} raw elements. Processing sequentially... ---")

        table_elements, table_contexts = [], []
        text_elements, text_contexts = [], []

        current_section_title = "Document Introduction"

        for el in elements:
            element_text = el.text.strip()
            if JUNK_FOOTER_PATTERN.match(element_text) or SEC_LINK_PATTERN.match(element_text):
                continue

            if isinstance(el, Title):
                title_text = el.text.strip()

                is_good_title = (
                        len(title_text) > 4 and
                        not JUNK_FOOTER_PATTERN.match(title_text) and
                        not CHECKBOX_PATTERN.search(title_text)
                )

                if is_good_title:
                    current_section_title = title_text

                continue

            eltype = "Table" if isinstance(el, Table) else "Text"

            contextt = Context(
                pdf_title=pdf_file_path.stem,
                page_number=getattr(el.metadata, 'page_number', None),
                section_title=current_section_title,
                element_type="Table" if isinstance(el, Table) else "Text"
            )

            # Step 2: Now, use the element AND the new context to generate keywords.
            keywords = get_relevant_keywords(el)

            # Step 3: Add the generated keywords back into the context object.
            contextt.relevant_keywords = keywords

            if isinstance(el, Table):
                if not _is_table_functionally_empty(getattr(el.metadata, 'text_as_html', None)):
                    table_elements.append(el)
                    table_contexts.append(contextt)
            else:
                if len(el.text.strip()) > 25:
                    text_elements.append(el)
                    text_contexts.append(contextt)

        print(
            f"--- Finished processing. Found {len(table_elements)} tables and {len(text_elements)} text elements. ---")

        return (table_elements, table_contexts), (text_elements, text_contexts)

    def _process_table_elements(self, table_elements: List, pdf_file_path: pathlib.Path, company_ticker: str) -> List[
        Document]:
        """
        Processes a list of table elements into Document objects.
        """
        table_chunks = []
        for table_el in table_elements:
            table_text = getattr(table_el.metadata, 'text_as_html', None) or table_el.text
            cleaned_text = _clean_table_text(table_text)

            context = Context(
                pdf_title=pdf_file_path.stem,
                page_number=1,
                section_title="abc123",
                relevant_keywords=get_relevant_keywords()
            )

            new_chunk = Document(
                page_content=cleaned_text,
                metadata={
                    "source": pdf_file_path.name,
                    "page": table_el.metadata.page_number,
                    "company": company_ticker,
                    "element_type": "table"
                }
            )
            table_chunks.append(new_chunk)
        return table_chunks

    def _process_text_elements(self, text_elements: List, pdf_file_path: pathlib.Path, company_ticker: str) -> List[
        Document]:
        """
        Chunks a list of text elements and processes them into Document objects.
        """
        text_chunks_raw = chunk_by_title(
            text_elements,
            max_characters=1000,
            new_after_n_chars=800,
            combine_text_under_n_chars=500,
            overlap=200,
            overlap_all=True
        )

        final_text_chunks = []
        for chunk in text_chunks_raw:
            cleaned_text = _clean_element_text(chunk.text)

            # Filter out very short, likely irrelevant, text snippets
            if len(cleaned_text) > 50:
                new_chunk = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": pdf_file_path.name,
                        "page": chunk.metadata.page_number,
                        "company": company_ticker,
                        "element_type": "text"
                    }
                )
                final_text_chunks.append(new_chunk)
        return final_text_chunks

    # !!! Main Function !!!
    def _load_pdf(self, pdf_file_path: pathlib.Path) -> List[Document]:
        """
        Loads, partitions, cleans, and chunks a PDF file into a list of Documents.
        """
        (table_elements, _), (text_elements, _) = self.partition_and_separate_elements(pdf_file_path)

        if not table_elements and not text_elements:
            return []

        company_ticker = pdf_file_path.stem.split('-')[0].upper()

        table_chunks = self._process_table_elements(table_elements, pdf_file_path, company_ticker)
        text_chunks = self._process_text_elements(text_elements, pdf_file_path, company_ticker)

        final_chunks = table_chunks + text_chunks

        print(f"--- Finished processing. Generated {len(final_chunks)} total chunks for {pdf_file_path.name}. ---\n")

        for doc_chunk in final_chunks:
            doc_chunk.page_content = html.unescape(doc_chunk.page_content)

        return final_chunks

    def load(self) -> List[Document]:
        return self._load_pdf(self.file_path)


if __name__ == "__main__":
    # --- Configuration ---
    # Define the path to the PDF you want to test.
    # Make sure this file exists in the same directory or provide a full path.
    test_file_path = "/Users/lukem/PycharmProjects/FinQuery2/reports/aapl-20230930.pdf"

    # --- Execution ---
    # Create an instance of your loader.
    # Note: We are calling a "private" method (_partition_and_separate_elements)
    # for direct testing, which is perfectly fine for a debug script like this.
    loader = CustomPDFLoader(file_path=test_file_path)

    try:
        (tables, table_contexts), (texts, text_contexts) = loader.partition_and_separate_elements(loader.file_path)

        # --- Output Verification ---
        print("\n\n" + "=" * 80)
        print("||" + " VERIFYING TABLE CHUNKS ".center(76) + "||")
        print("=" * 80)

        # Zip the tables and their contexts together and iterate through them
        for table, context in zip(tables, table_contexts):
            print("\n--- TABLE ELEMENT ---")
            # Use the _to_string() method you defined for a clean context printout
            print(context.to_string())

            # Print the actual content of the table (as HTML)
            print("[CONTENT]")
            table_html = htmltabletomd.convert_table(html.unescape(getattr(table.metadata, 'text_as_html', table.text)))
            print(table_html[:1000] + "..." if len(table_html) > 1000 else table_html)  # Truncate long tables
            print("-" * 50)

        print("\n\n" + "=" * 80)
        print("||" + " VERIFYING TEXT CHUNKS ".center(76) + "||")
        print("=" * 80)

        # Zip the text elements and their contexts together
        for text, context in zip(texts, text_contexts):
            # We can add a simple filter to ignore very small, likely noisy text chunks
            if len(text.text) < 50:
                continue

            print("\n--- TEXT ELEMENT ---")
            print(context.to_string())

            print("[CONTENT]")
            print(text.text)
            print("-" * 50)

    except FileNotFoundError:
        print(f"ERROR: The file '{test_file_path}' was not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
