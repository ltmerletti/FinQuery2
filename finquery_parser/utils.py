# --- Std Lib Imports ---
import html
import os
import pathlib
import re
from typing import Dict, List, Optional, Set, Tuple

# --- Special Imports ---
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
    pdf_file_path: pathlib.Path,
    junk_filter_patterns: Optional[List[re.Pattern]] = None,
    title_exclude_patterns: Optional[List[re.Pattern]] = None,
    custom_stop_words: Optional[Set[str]] = None,
    max_keywords: int = 5
) -> Tuple[Tuple[List[Table], List[Context]], Tuple[List[Text], List[Context]]]:
    """
    Partitions a PDF, enriches elements with metadata, and separates them.

    Args:
        pdf_file_path: The path pointing to the PDF file.
        junk_filter_patterns: An optional list of compiled regex patterns. Text
            elements matching any of these patterns will be discarded. Defaults
            to the standard JUNK_FOOTER_PATTERN and SEC_LINK_PATTERN.
        title_exclude_patterns: An optional list of compiled regex patterns.
            Titles matching any of these will not be used as section titles.
            Defaults to the standard CHECKBOX_PATTERN.
        custom_stop_words: An optional set of strings to be added to the default
            stop words list for keyword extraction.
        max_keywords: The maximum number of keywords to extract for each element's
            context. Defaults to 5.

    Returns:
        A tuple containing two tuples:
        1. A tuple of (list of `Table` elements, list of corresponding `Context` objects).
        2. A tuple of (list of `Text` elements, list of corresponding `Context` objects).
    """
    # --- Set default patterns if none are provided ---
    if junk_filter_patterns is None:
        junk_filter_patterns = [JUNK_FOOTER_PATTERN, SEC_LINK_PATTERN]
    if title_exclude_patterns is None:
        title_exclude_patterns = [CHECKBOX_PATTERN]

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
        # --- Use the provided junk filter patterns ---
        if any(pattern.match(element_text) for pattern in junk_filter_patterns):
            continue

        if isinstance(el, Title):
            title_text = el.text.strip()
            # --- Check against both default and custom exclusion patterns ---
            is_good_title = (
                len(title_text) > 4 and
                not any(p.match(title_text) for p in junk_filter_patterns) and
                not any(p.search(title_text) for p in title_exclude_patterns)
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
        # --- Pass custom stop words to the keyword extraction function ---
        context.relevant_keywords = get_relevant_keywords(
            el, context, max_keywords=max_keywords, custom_stop_words=custom_stop_words
        )

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


def load_pdf(
    pdf_file_path: pathlib.Path,
    **kwargs
) -> List[Document]:
    """
    Loads a PDF, partitions it, and converts elements into Document objects.

    This function acts as a wrapper and passes any additional keyword arguments
    (e.g., `junk_filter_patterns`, `custom_stop_words`) directly to the
    `partition_and_separate_elements` function, allowing for flexible processing.

    Args:
        pdf_file_path: The `pathlib.Path` object for the PDF to load.
        **kwargs: Additional keyword arguments to pass to the partitioner, such as:
            - `junk_filter_patterns`: Optional list of regex patterns to filter junk.
            - `title_exclude_patterns`: Optional list of regex patterns for titles.
            - `custom_stop_words`: Optional set of custom stop words.
            - `max_keywords`: Integer for the max number of keywords.

    Returns:
        A list of `Document` objects for text and tables from the PDF.
    """
    (table_elements, table_contexts), (text_elements, text_contexts) = \
        partition_and_separate_elements(pdf_file_path, **kwargs)

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

def clean_element_text(
    text: str,
    cleaning_rules: Optional[List[Tuple[str, str]]] = None
) -> str:
    """
    Applies regex substitutions to clean textual content.

    Allows for custom cleaning rules. If none are provided, it uses a default
    set of rules to remove URLs, dates, and other common artifacts.

    Args:
        text: The raw string content from a text element.
        cleaning_rules: An optional list of (pattern, replacement) tuples.
            The pattern is a regex string, and replacement is its substitute.

    Returns:
        A cleaned string.
    """
    if cleaning_rules is None:
        # Default rules if none are provided, ensuring backward compatibility
        cleaning_rules = [
            (r'https?://\S+', ''),
            (r'\S*www\.sec\.gov\S*', ''),
            (r'\s*\d+/\d+\s*', ''),
            (r'\d{1,2}/\d{1,2}/\d{2,4}(,\s*\d{1,2}:\d{2}\s*(AM|PM)?)?', ''),
            (r'^\s*\d+\s*$', ''),
        ]

    for pattern, replacement in cleaning_rules:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    text = re.sub(r'\n\s*\n', '\n', text) # Consolidate multiple newlines
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

    master_prompt_template = """You are an ultra-precise API endpoint named 'JsonFinSummarizer'. Your only function is to receive a financial table and return a single, clean JSON object. You do not provide any explanation, preamble, or conversational text.

**JSON OUTPUT SPECIFICATION:**
Your output MUST be a valid JSON object containing a single key called "summary". The value of the "summary" key MUST be a single, descriptive sentence.

**CONTENT RULES FOR THE SUMMARY SENTENCE:**
1.  **DO NOT** include any specific numbers, dollar amounts, or percentages from the table's data cells.
2.  **DO** state the main subject of the table (e.g., Net Sales, Assets and Liabilities).
3.  **DO** state the primary dimensions or categories (e.g., by Product Category, by Geographic Segment).
4.  **DO** state the time period if available (e.g., for fiscal years 2021-2023).
5.  Your entire response must be ONLY the JSON object, with no leading/trailing characters, newlines, or markdown code fences.

**EXAMPLES:**

**Example 1:**
---
[USER]
Section: Products and Services Performance
Table:
| Category | 2023 | 2022 |
| :--- | :--- | :--- |
| iPhone | $200,583 | $205,489 |
| Mac | $29,357 | $40,177 |
| Services | $85,200 | $78,129 |

[ASSISTANT]
{{"summary": "A breakdown of net sales by product category, including iPhone, Mac, and Services, for fiscal years 2022 and 2023."}}
---

**Example 2:**
---
[USER]
Section: CONSOLIDATED BALANCE SHEETS
Table:
| | 2023 | 2022 |
| :--- | :--- | :--- |
| Total assets | 352,583 | 352,755 |
| Total liabilities | 290,437 | 302,083 |

[ASSISTANT]
{{"summary": "A consolidated balance sheet comparing total assets and total liabilities between fiscal years 2023 and 2022."}}
---
\\no_think"""
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
    element: Element,
    context: Context,
    max_keywords: int = 15,
    custom_stop_words: Optional[Set[str]] = None,
    keyword_patterns: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Extracts relevant keywords from a document element using heuristics.

    This version allows for adding custom stop words and overriding the default
    regex patterns used to find keyword candidates.

    Args:
        element: The `unstructured` `Element` (e.g., `Table` or `Text`).
        context: The `Context` object associated with the element.
        max_keywords: The maximum number of keywords to return.
        custom_stop_words: An optional set of strings to add to the default
            stop words list.
        keyword_patterns: An optional dictionary to override default regex.
            Expected keys: 'capitalized_phrases', 'acronyms'.

    Returns:
        A list of cleaned, relevant keyword strings.
    """
    if not element or not element.text.strip():
        return []

    # --- Define default regex patterns for keyword extraction ---
    default_patterns = {
        'capitalized_phrases': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b',
        'acronyms': r'\b[A-Z]{2,5}\b'
    }
    if keyword_patterns:
        default_patterns.update(keyword_patterns)

    # --- Combine default and custom stop words ---
    all_stop_words = STOP_WORDS.copy()
    if custom_stop_words:
        all_stop_words.update(s.lower() for s in custom_stop_words)

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

    # --- Use configurable regex patterns ---
    capitalized_phrases = re.findall(default_patterns['capitalized_phrases'], text_to_process)
    candidates.update(capitalized_phrases)
    acronyms = re.findall(default_patterns['acronyms'], text_to_process)
    candidates.update(acronyms)

    final_keywords = []
    seen_lower = set()
    sorted_candidates = sorted(list(c for c in candidates if isinstance(c, str)), key=len, reverse=True)

    for keyword in sorted_candidates:
        kw_clean = keyword.strip(" “”)’'.,:()").replace('’', "'")
        kw_lower = kw_clean.lower()

        if not kw_clean or len(kw_clean) <= 3 or len(kw_clean) >= 30 or kw_lower in all_stop_words or kw_lower.isdigit():
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