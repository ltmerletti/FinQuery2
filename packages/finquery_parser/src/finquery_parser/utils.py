# Standard lib imports
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Tokenizers
import tiktoken

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import (DOCLING_LAYOUT_EGRET_LARGE, )
from docling.datamodel.pipeline_options import PdfPipelineOptions, LayoutOptions
from docling.datamodel.pipeline_options import (TableFormerMode, TableStructureOptions, )
from docling.document_converter import DocumentConverter, PdfFormatOption

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from finquery_app.config import LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, LMSTUDIO_MODEL_NAME

# Local imports
from finquery_parser.types import *
from finquery_parser.types import _DoclingElementAdapter

# ==============================================================================
#  Constants and Configuration
# ==============================================================================
MAX_CHUNK_TOKENS = 350

# This pattern is designed to remove footer text that often appears in SEC filings.
JUNK_FOOTER_PATTERN = re.compile(r'^.*Form 10-K\s*\|\s*\d+\s*$', re.IGNORECASE | re.MULTILINE)

# This pattern filters out hyperlinks to SEC filings.
SEC_LINK_PATTERN = re.compile(r'https?://www\.sec\.gov/Archives/edgar/data/.*\.htm', re.IGNORECASE)

# A comprehensive set of stop words to ensure extracted keywords are meaningful.
STOP_WORDS = {'a', 'an', 'and', 'the', 'is', 'it', 'in', 'on', 'for', 'of', 'as', 'to', 'inc', 'was', 'were', 'by',
              'with', 'or', 'at', 'from', 'that', 'this', 'llc', 'ltd', 'company', 'corp', 'about', 'after', 'all',
              'also', 'been', 'because', 'but', 'can', 'could', 'did', 'do', 'due', 'has', 'had', 'have', 'how',
              'however', 'into', 'its', 'just', 'may', 'most', 'must', 'not', 'other', 'our', 'out', 'over', 'said',
              'should', 'so', 'some', 'such', 'than', 'then', 'there', 'these', 'they', 'through', 'under', 'upon',
              'use', 'used', 'using', 'various', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
              'who', 'why', 'will', 'would', 'you', 'your', 'notes', 'note', 'see', 'title', 'part', 'item', 'items',
              'page', 'inc', 'corp', 'ltd', 'llc', 'the company', 'apple inc', 'registrant', 'thereof', 'thereto',
              'therein', 'thereon', 'hereto', 'hereof', 'herein', 'hereinafter', 'pursuant', 'including', 'certain',
              'related', 'primarily', 'approximately', 'significant', 'generally'}


# ==============================================================================
#  Core Processing Functions
# ==============================================================================
def _build_page_to_section_map(doc: DoclingDoc) -> Dict[int, str]:
    """
    Scans the document, including headers/footers (furniture), to build a
    mapping from page number to the corresponding section title.
    """
    page_to_section = {}
    # Pattern made to match financial documents (where things like item/part are common)
    section_title_pattern = re.compile(r'^\s*(ITEM|PART)\s+[\dIVX]+\.?\s*', re.IGNORECASE)
    all_text_items = doc.texts
    # Sort text items by page and also the bounding box coordinates (bbox)
    all_text_items.sort(key=lambda text_item: (text_item.prov[0].page_no if text_item.prov else 0,
                                               -text_item.prov[0].bbox.t if text_item.prov and text_item.prov[
                                                   0].bbox else 0))

    # Check for section titles on pages
    current_section_by_page = {}
    for item in all_text_items:
        page_no = item.prov[0].page_no if item.prov else None
        if page_no is None or page_no in current_section_by_page:
            continue

        text = item.text.strip()
        if section_title_pattern.match(text):
            clean_title = section_title_pattern.sub('', text).strip()
            if len(clean_title) > 3:
                current_section_by_page[page_no] = clean_title

    max_page = max(p.page_no for p in doc.pages.values()) if doc.pages else 0
    last_title = "Document Introduction"
    for i in range(1, max_page + 2):
        if i in current_section_by_page:
            last_title = current_section_by_page[i]
        page_to_section[i] = last_title

    return page_to_section


def partition_and_separate_elements(pdf_file_path: pathlib.Path,
                                    junk_filter_patterns: Optional[List[re.Pattern]] = None,
                                    custom_stop_words: Optional[Set[str]] = None, max_keywords: int = 5,
                                    llm: ChatOpenAI = None, use_high_res: Optional[bool] = False) -> Tuple[
    Tuple[List[DoclingTableAdapter], List[Context]], Tuple[List[DoclingTextAdapter], List[Context]]]:
    """
    Partitions a PDF using Docling, enriches elements with metadata, and separates them.
    This version correctly handles section titles and attaches preceding text to tables.
    """
    if junk_filter_patterns is None:
        junk_filter_patterns = [JUNK_FOOTER_PATTERN, SEC_LINK_PATTERN]

    print(f"Partitioning document with Docling: {pdf_file_path}")
    if not str(pdf_file_path).endswith(".pdf"):
        return ([], []), ([], [])

    try:
        if use_high_res:
            pipeline_options = PdfPipelineOptions(do_table_structure=True,
                                                  table_structure_options=TableStructureOptions(
                                                      mode=TableFormerMode.ACCURATE, do_cell_matching=False))
        else:
            pipeline_options = PdfPipelineOptions(do_table_structure=True,
                                                  layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_LARGE),
                                                  table_structure_options=TableStructureOptions(
                                                      mode=TableFormerMode.ACCURATE, do_cell_matching=False))
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        doc = converter.convert(pdf_file_path).document
        print(f"\n--- Docling conversion complete. Building section map... ---")
    except Exception as e:
        print(f"Docling conversion failed for {pdf_file_path}: {e}")
        return ([], []), ([], [])

    page_to_section_map = _build_page_to_section_map(doc)
    all_elements = list(doc.iterate_items())
    processed_tables = []
    consumed_text_ids = set()

    for i, (item, _) in enumerate(all_elements):
        if isinstance(item, DoclingTableItem):
            table_adapter = DoclingTableAdapter(item, doc)
            if not table_adapter.text.strip():
                continue

            table_prefix = ""
            if i > 0:
                prev_item, _ = all_elements[i - 1]
                if isinstance(prev_item, DoclingTextItem):
                    if 0 < len(prev_item.text.strip()) < 150:
                        table_prefix = prev_item.text.strip()
                        consumed_text_ids.add(id(prev_item))

            processed_tables.append({'adapter': table_adapter, 'prefix': table_prefix})

    final_table_contexts = []
    final_text_elements_with_context = []
    #
    table_map = {id(t['adapter']._item): t for t in processed_tables}

    for item, _ in all_elements:
        # Get page number and section title
        page_no = item.prov[0].page_no if item.prov and item.prov[0] else 1
        section_title = page_to_section_map.get(page_no, "Document Introduction")
        item_id = id(item)

        # If it's a docling table item:
        if isinstance(item, DoclingTableItem) and item_id in table_map:
            table_info = table_map[item_id]
            adapter = table_info['adapter']
            # Create context
            context = Context(pdf_file_path.stem, page_no, section_title, "Table", "",
                              table_prefix=table_info['prefix'])
            context.relevant_keywords = get_relevant_keywords(adapter, context, max_keywords=max_keywords,
                                                              custom_stop_words=custom_stop_words)
            final_table_contexts.append(context)

        # Otherwise if it's a text item
        elif isinstance(item, DoclingTextItem) and item_id not in consumed_text_ids and item.label != 'heading':
            adapter = DoclingTextAdapter(item, doc)
            element_text = adapter.text.strip()
            # Make sure it's not a junk item
            if len(element_text) > 25 and not any(pattern.search(element_text) for pattern in junk_filter_patterns):
                context = Context(pdf_file_path.stem, page_no, section_title, "Text", "")
                context.relevant_keywords = get_relevant_keywords(adapter, context, max_keywords=max_keywords,
                                                                  custom_stop_words=custom_stop_words)
                final_text_elements_with_context.append((adapter, context))

    final_table_elements = [t['adapter'] for t in processed_tables]
    final_text_elements = [item[0] for item in final_text_elements_with_context]
    final_text_contexts = [item[1] for item in final_text_elements_with_context]

    print(
        f"--- Finished processing. Found {len(final_table_elements)} tables and {len(final_text_elements)} text elements. ---")
    return (final_table_elements, final_table_contexts), (final_text_elements, final_text_contexts)

def load_pdf(pdf_file_path: pathlib.Path, llm: ChatOpenAI, **kwargs) -> List[Document]:
    """
    Loads a PDF, partitions it, and converts elements into intelligently chunked Document objects.
    Tables are returned as separate documents, and text is chunked using an intelligent grouping
    strategy to preserve semantic cohesion.
    """
    (table_elements, table_contexts), (text_elements, text_contexts) = partition_and_separate_elements(
        pdf_file_path,
        llm=llm,
        **kwargs
    )

    # If the initial pass finds no tables, rerun the partitioning once in high-resolution mode
    # This redundancy is intentional to overcome potential intermittent detection failures
    # Sometimes the computer falling to sleep or other issues can cause the parsing to not detect tables
    if not table_elements:
        print("No tables detected on initial pass. Retrying in high-resolution mode...")
        (table_elements, table_contexts), (text_elements, text_contexts) = partition_and_separate_elements(
            pdf_file_path,
            llm=llm,
            use_high_res=True
        )

    print(f"Number of tables: {len(table_elements)}")

    if not table_elements and not text_elements:
        return []

    company_ticker = pdf_file_path.stem.split('-')[0].upper()
    final_chunks: List[Document] = []
    parser = define_parser()

    print(f"Augmenting and assembling {len(table_elements)} table chunks...")
    for table, context in zip(table_elements, table_contexts):
        try:
            summary = get_one_line_summary(table.text, context.section_title, parser, llm)
            context.summary = summary.get('summary', "Error: Summary key missing.")
        except TableSummarizationError as e:
            print(e)
            context.summary = "Error: Table summarization failed."

        context_string = context.to_string()
        md_table = table.text

        content_string = f"{context.table_prefix}\n\n{md_table}" if context.table_prefix else md_table

        augmented_content = f"{context_string}\n\n[CONTENT]\n{content_string.strip()}"
        new_doc = Document(page_content=augmented_content,
                           metadata={"source": pdf_file_path.name, "page": context.page_number,
                                     "company": company_ticker, "element_type": "Table",
                                     "section": context.section_title,
                                     "keywords": ", ".join(context.relevant_keywords)})
        final_chunks.append(new_doc)

    print(f"Grouping and chunking {len(text_elements)} text elements by section...")

    tokenizer = tiktoken.get_encoding("cl100k_base")

    section_texts = defaultdict(list)
    for text_element, context in zip(text_elements, text_contexts):
        section_texts[context.section_title].append((text_element.text, context))

    # First loop, going through the section texts
    for section_title, content_list in section_texts.items():
        current_chunk_texts = []
        current_chunk_contexts = []
        current_token_count = 0

        # Loop through the text within
        for text, context in content_list:
            # Clean the text
            cleaned_text = clean_element_text(text)
            if not cleaned_text:
                continue

            # Split the text. Tables are left whole.
            element_tokens = len(tokenizer.encode(cleaned_text))
            if element_tokens > MAX_CHUNK_TOKENS:
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base",
                                                                                     chunk_size=MAX_CHUNK_TOKENS,
                                                                                     chunk_overlap=100)
                sub_chunks = text_splitter.split_text(cleaned_text)
                for sub_chunk in sub_chunks:
                    context_string = context.to_string()
                    augmented_content = f"{context_string}\n\n[CONTENT]\n{sub_chunk}"
                    final_chunks.append(Document(page_content=augmented_content,
                                                 metadata={"source": pdf_file_path.name, "page": context.page_number,
                                                           "company": company_ticker, "element_type": "Text",
                                                           "section": section_title,
                                                           "keywords": ", ".join(context.relevant_keywords)}))
                continue

            # If this would go over the token count manually split it instead
            if current_token_count + element_tokens > MAX_CHUNK_TOKENS and current_chunk_texts:
                final_text = "\n\n".join(current_chunk_texts)
                first_context = current_chunk_contexts[0]

                aggregated_keywords = sorted(
                    list(set(kw for ctx in current_chunk_contexts for kw in ctx.relevant_keywords)))
                first_context.relevant_keywords = aggregated_keywords

                context_string = first_context.to_string()
                augmented_content = f"{context_string}\n\n[CONTENT]\n{final_text}"

                final_chunks.append(Document(page_content=augmented_content,
                                             metadata={"source": pdf_file_path.name, "page": first_context.page_number,
                                                       "company": company_ticker, "element_type": "Text",
                                                       "section": section_title,
                                                       "keywords": ", ".join(aggregated_keywords)}))

                current_chunk_texts = [cleaned_text]
                current_chunk_contexts = [context]
                current_token_count = element_tokens
            else:
                current_chunk_texts.append(cleaned_text)
                current_chunk_contexts.append(context)
                current_token_count += element_tokens

        if current_chunk_texts:
            final_text = "\n\n".join(current_chunk_texts)
            first_context = current_chunk_contexts[0]
            aggregated_keywords = sorted(
                list(set(kw for ctx in current_chunk_contexts for kw in ctx.relevant_keywords)))
            first_context.relevant_keywords = aggregated_keywords

            context_string = first_context.to_string()
            augmented_content = f"{context_string}\n\n[CONTENT]\n{final_text}"

            final_chunks.append(Document(page_content=augmented_content,
                                         metadata={"source": pdf_file_path.name, "page": first_context.page_number,
                                                   "company": company_ticker, "element_type": "Text",
                                                   "section": section_title,
                                                   "keywords": ", ".join(aggregated_keywords)}))

    return final_chunks


# ==============================================================================
#  Helper and Utility Functions
# ==============================================================================

def clean_element_text(text: str, cleaning_rules: Optional[List[Tuple[str, str]]] = None) -> str:
    """
    Applies regex substitutions to clean textual content.
    """
    if cleaning_rules is None:
        cleaning_rules = [  # Remove HTTP/HTTPS URLs like https://example.com/page
            (r'https?://\S+', ''),

            # Remove links to sec.gov
            (r'\S*www\.sec\.gov\S*', ''),

            # Remove simple fractions (page numbers)
            (r'\s*\d+/\d+\s*', ''),

            # Remove full dates (PDF artifact)
            (r'\d{1,2}/\d{1,2}/\d{2,4}(,\s*\d{1,2}:\d{2}\s*(AM|PM)?)?', ''),

            # Remove lines that contain only numbers
            (r'^\s*\d+\s*$', ''), ]

    for pattern, replacement in cleaning_rules:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    text = re.sub(r'\n\s*\n', '\n', text)  # Consolidate multiple newlines
    return text.strip()


def define_parser() -> JsonOutputParser:
    """Initializes a JSON output parser for a specific Pydantic model."""
    return JsonOutputParser(pydantic_object=TableSummary)


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from the response"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def get_one_line_summary(table_text: str, section_title: str, parser: JsonOutputParser, llm: ChatOpenAI) -> Dict:
    """Generates a one-sentence summary of a financial table using an LLM."""
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
        [("system", master_prompt_template), ("human", "Section: {section_title}\n\nTable:\n{table}")])
    chain = prompt | llm | (lambda x: strip_thinking_tags(x.content)) | parser

    try:
        result = chain.invoke({"table": table_text, "section_title": section_title})
    except Exception as e:
        raise TableSummarizationError(f"LLM call failed for table in section '{section_title}'") from e

    return result


def get_relevant_keywords(element: _DoclingElementAdapter, context: Context, max_keywords: int = 15,
                          custom_stop_words: Optional[Set[str]] = None,
                          keyword_patterns: Optional[Dict[str, str]] = None) -> List[str]:
    """Extracts relevant keywords from a document element using heuristics."""
    if not hasattr(element, 'text') or not element.text.strip():
        return []

    default_patterns = {'capitalized_phrases': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', 'acronyms': r'\b[A-Z]{2,5}\b'}
    if keyword_patterns:
        default_patterns.update(keyword_patterns)

    all_stop_words = STOP_WORDS.copy()
    if custom_stop_words:
        all_stop_words.update(s.lower() for s in custom_stop_words)

    text_to_process = element.text
    candidates = set()

    if isinstance(element, DoclingTableAdapter):
        table_markdown = element.text
        if table_markdown:
            lines = table_markdown.strip().split('\n')
            if lines:
                header_line = lines[0]
                headers = [h.strip() for h in header_line.split('|') if h.strip()]
                candidates.update(headers)

    if context.section_title:
        candidates.add(context.section_title)

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

        if not kw_clean or len(kw_clean) <= 3 or len(
                kw_clean) >= 30 or kw_lower in all_stop_words or kw_lower.isdigit():
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

# ==============================================================================
#  Main Execution Block - For Debugging and Testing
# ==============================================================================
def main():
    """
    Main function to execute the PDF loading and processing pipeline.
    """
    pdf_to_process = pathlib.Path("/reports/pltr-20231231.pdf")

    llm_base_url = LMSTUDIO_BASE_URL
    llm_api_key = LMSTUDIO_API_KEY
    llm_model_name = LMSTUDIO_MODEL_NAME

    print(f"Initializing LLM with base URL: {llm_base_url} and model: {llm_model_name}")

    llm = ChatOpenAI(model=llm_model_name, api_key=llm_api_key, base_url=llm_base_url, temperature=0.1)

    print(f"Starting the loading process for: {pdf_to_process.name}\n")

    documents = load_pdf(pdf_to_process, llm=llm, use_high_res=False)

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
        print(f"Page:     {doc.metadata.get('page', 'N/A')}")
        print(f"Section:  {doc.metadata.get('section', 'N/A')}")
        print(f"Keywords: {doc.metadata.get('keywords', 'N/A')}")
        print("\n--- Content ---\n")
        print(doc.page_content)
        print("\n-----------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()