"""
This module provides a suite of tools for parsing, cleaning, and structuring
content from financial PDF documents, specifically focusing on extracting and
enriching text and table elements for AI-driven analysis.
"""

# --- Standard Library Imports ---
import html
import os
import pathlib
import re
from typing import Dict, List, Tuple

# --- Third-Party Imports ---
import bs4
import htmltabletomd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from unstructured.documents.elements import Element, Table, Text, Title
from unstructured.partition.pdf import partition_pdf

# --- Local Imports ---
from .types import Context, TableSummary

# ==============================================================================
#  Custom Exception
# ==============================================================================

class TableSummarizationError(Exception):
    """Custom exception for errors during the LLM table summarization step."""
    pass

# ==============================================================================
#  Constants and Configuration
# ==============================================================================

# This pattern is designed to remove footer text that often appears in SEC filings.
JUNK_FOOTER_PATTERN = re.compile(
    r'^.*Form 10-K\s*\|\s*\d+\s*$', re.IGNORECASE | re.MULTILINE
)

# This pattern helps identify and discard titles that are just form-like checkboxes.
CHECKBOX_PATTERN = re.compile(r'Yes\s*☒\s*No\s*☐', re.IGNORECASE)

# This pattern filters out hyperlinks to SEC filings.
SEC_LINK_PATTERN = re.compile(
    r'https?://www\.sec\.gov/Archives/edgar/data/.*\.htm', re.IGNORECASE
)

# A comprehensive set of stop words to ensure extracted keywords are meaningful.
STOP_WORDS = {
    'a', 'an', 'and', 'the', 'is', 'it', 'in', 'on', 'for', 'of', 'as', 'to',
    'inc', 'was', 'were', 'by', 'with', 'or', 'at', 'from', 'that', 'this',
    'llc', 'ltd', 'company', 'corp', 'about', 'after', 'all', 'also',
    'been', 'because', 'but', 'can', 'could', 'did', 'do', 'due', 'has',
    'had', 'have', 'how', 'however', 'into', 'its', 'just', 'may', 'most',
    'must', 'not', 'other', 'our', 'out', 'over', 'said', 'should', 'so',
    'some', 'such', 'than', 'then', 'there', 'these', 'they', 'through',
    'under', 'upon', 'use', 'used', 'using', 'various', 'very', 'was',
    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'why',
    'will', 'would', 'you', 'your', 'notes', 'note', 'see', 'title',
    'part', 'item', 'items', 'page', 'inc', 'corp', 'ltd', 'llc',
    'the company', 'apple inc', 'registrant', 'thereof', 'thereto', 'therein',
    'thereon', 'hereto', 'hereof', 'herein', 'hereinafter', 'pursuant',
    'including', 'certain', 'related', 'primarily', 'approximately',
    'significant', 'generally'
}


# ==============================================================================
#  Core Processing Functions
# ==============================================================================

def partition_and_separate_elements(
    pdf_file_path: pathlib.Path
) -> Tuple[Tuple[List[Table], List[Context]], Tuple[List[Text], List[Context]]]:
    """
    Partitions a PDF document into its constituent elements, enriches them with
    contextual metadata, and separates them into tables and text.

    Args:
        pdf_file_path: The `pathlib.Path` object pointing to the PDF file.

    Returns:
        A tuple containing two tuples:
        1. A tuple of (list of `Table` elements, list of corresponding `Context` objects).
        2. A tuple of (list of `Text` elements, list of corresponding `Context` objects).
    """
    print(f"Partitioning document: {pdf_file_path}")
    if not str(pdf_file_path).endswith(".pdf"):
        return ([], []), ([], [])

    elements = list(partition_pdf(
        pdf_file_path, strategy="hi_res", infer_table_structure=True
    ))
    print(f"\n--- Found {len(elements)} raw elements. Processing sequentially... ---")

    table_elements, table_contexts = [], []
    text_elements, text_contexts = [], []
    current_section_title = "Document Introduction"
    parser = define_parser()

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

        context = Context(
            pdf_title=pdf_file_path.stem,
            page_number=getattr(el.metadata, 'page_number', None),
            section_title=current_section_title,
            element_type="Table" if isinstance(el, Table) else "Text",
            summary=""
        )
        context.relevant_keywords = get_relevant_keywords(el, context, 5)

        if context.element_type == "Table":
            try:
                one_sentence_summary = get_one_line_summary(el.text, context.section_title, parser)
                context.summary = one_sentence_summary.get('summary', "Error: Summary key missing.")
            except TableSummarizationError as e:
                print(e) # Log the specific error
                context.summary = "Error: Table summarization failed."


        if isinstance(el, Table):
            if not is_table_functionally_empty(getattr(el.metadata, 'text_as_html', None)):
                table_elements.append(el)
                table_contexts.append(context)
        else:
            if len(el.text.strip()) > 25:
                text_elements.append(el)
                text_contexts.append(context)

    print(f"--- Finished processing. Found {len(table_elements)} tables and {len(text_elements)} text elements. ---")
    return (table_elements, table_contexts), (text_elements, text_contexts)


def load_pdf(pdf_file_path: pathlib.Path) -> List[Document]:
    """
    Loads a PDF file, partitions it, and converts the structured elements
    into a list of `langchain_core.documents.Document` objects.

    Args:
        pdf_file_path: The `pathlib.Path` object for the PDF to load.

    Returns:
        A list of `Document` objects, each representing a chunk of text or a
        table from the PDF, augmented with contextual metadata.
    """
    (table_elements, table_contexts), (text_elements, text_contexts) = \
        partition_and_separate_elements(pdf_file_path)

    if not table_elements and not text_elements:
        return []

    company_ticker = pdf_file_path.stem.split('-')[0].upper()
    final_chunks: List[Document] = []

    print(f"Augmenting and assembling {len(table_elements)} table chunks...")
    for table, context in zip(table_elements, table_contexts):
        context_string = context.to_string()
        table_html = html.unescape(getattr(table.metadata, 'text_as_html', table.text))
        content_string = htmltabletomd.convert_table(table_html)
        augmented_content = f"{context_string}\n\n[CONTENT]\n{content_string}"
        new_doc = Document(
            page_content=augmented_content,
            metadata={
                "source": pdf_file_path.name,
                "page": context.page_number,
                "company": company_ticker,
                "element_type": "Table"
            }
        )
        final_chunks.append(new_doc)

    for text, context in zip(text_elements, text_contexts):
        context_string = context.to_string()
        content_string = clean_element_text(text.text)
        augmented_content = f"{context_string}\n\n[CONTENT]\n{content_string}"
        new_doc = Document(
            page_content=augmented_content,
            metadata={
                "source": pdf_file_path.name,
                "page": context.page_number,
                "company": company_ticker,
                "element_type": "Text"
            }
        )
        final_chunks.append(new_doc)

    return final_chunks


# ==============================================================================
#  Helper and Utility Functions
# ==============================================================================

def clean_element_text(text: str) -> str:
    """
    Applies a series of regex substitutions to clean textual content.

    Args:
        text: The raw string content from a text element.

    Returns:
        A cleaned string.
    """
    text = re.sub(r'https?://\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*www\.sec\.gov\S*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\d+/\d+\s*', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}(,\s*\d{1,2}:\d{2}\s*(AM|PM)?)?', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def clean_table_text(text: str) -> str:
    """
    Performs minimal cleaning on table text.

    Args:
        text: The raw string content from a table element.

    Returns:
        A cleaned string.
    """
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def define_parser() -> JsonOutputParser:
    """
    Initializes a JSON output parser for a specific Pydantic model.

    Returns:
        An instance of `JsonOutputParser` configured to parse JSON into the
        `TableSummary` Pydantic object.
    """
    return JsonOutputParser(pydantic_object=TableSummary)


def get_one_line_summary(
    table_text: str,
    section_title: str,
    parser: JsonOutputParser
) -> Dict:
    """
    Generates a one-sentence summary of a financial table using an LLM.

    Args:
        table_text: The string representation of the table.
        section_title: The title of the section containing the table.
        parser: The `JsonOutputParser` to process the LLM's response.

    Returns:
        A dictionary containing the summary, e.g., `{"summary": "..."}`.

    Raises:
        TableSummarizationError: If the LLM call or output parsing fails.
    """
    load_dotenv()
    llm = ChatOpenAI(
        model=os.getenv("LMSTUDIO_MODEL_NAME"),
        base_url=os.getenv("LMSTUDIO_BASE_URL"),
        api_key=os.getenv("LMSTUDIO_API_KEY")
    )

    master_prompt_template = """You are an ultra-precise API endpoint... (rest of prompt)""" # Kept for brevity
    prompt = ChatPromptTemplate.from_messages(
        [("system", master_prompt_template), ("human", "Section: {section_title}\n\nTable:\n{table}")]
    )
    chain = prompt | llm | parser

    # --- MODIFIED ERROR HANDLING ---
    try:
        result = chain.invoke({"table": table_text, "section_title": section_title})
    except Exception as e:
        # Raise a specific error that can be caught by the calling function,
        # preserving the original error for easier debugging.
        raise TableSummarizationError(
            f"LLM call failed for table in section '{section_title}'"
        ) from e

    return result


def get_relevant_keywords(
    element: Element, context: Context, max_keywords: int = 15
) -> List[str]:
    """
    Extracts high-quality, relevant keywords from a document element using
    a set of refined, zero-dependency heuristics.

    Args:
        element: The `unstructured` `Element` (e.g., `Table` or `Text`).
        context: The `Context` object associated with the element.
        max_keywords: The maximum number of keywords to return.

    Returns:
        A list of cleaned, relevant keyword strings.
    """
    if not element or not element.text.strip():
        return []

    text_to_process = element.text
    candidates = set()

    if isinstance(element, Table):
        table_html = getattr(element.metadata, 'text_as_html', '')
        if table_html:
            soup = bs4.BeautifulSoup(table_html, 'html.parser')
            for header in soup.find_all('th'):
                candidates.add(header.get_text(strip=True))
            if not soup.find_all('th'):
                for row in soup.find_all('tr'):
                    first_cell = row.find('td')
                    if first_cell:
                        cell_text = first_cell.get_text(strip=True)
                        if not re.match(r'^\(?[$\d,.]+\)?$', cell_text):
                            candidates.add(cell_text)

    if context.section_title:
        candidates.add(context.section_title)
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text_to_process)
    candidates.update(capitalized_phrases)
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', text_to_process)
    candidates.update(acronyms)

    final_keywords = []
    seen_lower = set()
    sorted_candidates = sorted(list(c for c in candidates if isinstance(c, str)), key=len, reverse=True)

    for keyword in sorted_candidates:
        kw_clean = keyword.strip(" “”)’'.,:()").replace('’', "'")
        kw_lower = kw_clean.lower()

        if not kw_clean or len(kw_clean) <= 3 or len(kw_clean) >= 30 or kw_lower in STOP_WORDS or kw_lower.isdigit():
            continue

        is_redundant = any(kw_lower in seen for seen in seen_lower)
        if is_redundant:
            continue

        if kw_lower not in seen_lower:
            final_keywords.append(kw_clean)
            seen_lower.add(kw_lower)

        if len(final_keywords) >= max_keywords:
            break

    return final_keywords


def is_table_functionally_empty(table_html: str) -> bool:
    """
    Checks if an HTML table contains any actual data in its `<td>` cells.

    Args:
        table_html: A string containing the HTML of the table.

    Returns:
        `True` if the table has no content in its data cells, `False` otherwise.
    """
    if not table_html:
        return True
    soup = bs4.BeautifulSoup(table_html, 'html.parser')
    return not any(cell.get_text(strip=True) for cell in soup.find_all('td'))