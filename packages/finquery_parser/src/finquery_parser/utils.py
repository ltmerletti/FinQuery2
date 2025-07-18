# Standard Library Imports
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

# Tokenizers & NLP Tools
from spacy.language import Language
from spacy import load
from tiktoken.core import Encoding
from tiktoken import get_encoding

# Docling Imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import (DOCLING_LAYOUT_EGRET_LARGE, )
from docling.datamodel.pipeline_options import PdfPipelineOptions, LayoutOptions
from docling.datamodel.pipeline_options import (TableFormerMode, TableStructureOptions, )
from docling.document_converter import DocumentConverter, PdfFormatOption

# Langchain Imports
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Local Imports
from finquery_parser.types import *
from finquery_app.config import LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, LMSTUDIO_MODEL_NAME

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
              'related', 'primarily', 'approximately', 'significant', 'generally', 'thereunto', 'therewith'}

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
                                    llm: ChatOpenAI = None, use_high_res: Optional[bool] = False, nlp: Language = None,
                                    tokenizer: Encoding = None) -> Tuple[
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
    consumed_text_ids = set()

    table_items_to_process = []
    text_items_to_process = []

    table_map = {}
    for i, (item, _) in enumerate(all_elements):
        if isinstance(item, DoclingTableItem):
            table_adapter = DoclingTableAdapter(item, doc)
            if not table_adapter.text.strip():
                continue
            table_prefix = ""
            if i > 0:
                prev_item, _ = all_elements[i - 1]
                if isinstance(prev_item, DoclingTextItem) and 0 < len(prev_item.text.strip()) < 150:
                    table_prefix = prev_item.text.strip()
                    consumed_text_ids.add(id(prev_item))
            table_map[id(item)] = {'adapter': table_adapter, 'prefix': table_prefix}

    for item, _ in all_elements:
        page_no = item.prov[0].page_no if item.prov and item.prov[0] else 1
        section_title = page_to_section_map.get(page_no, "Document Introduction")
        item_id = id(item)

        if isinstance(item, DoclingTableItem) and item_id in table_map:
            table_info = table_map[item_id]
            context = Context(pdf_file_path.stem, page_no, section_title, "Table", "",
                              table_prefix=table_info['prefix'])
            table_items_to_process.append({'adapter': table_info['adapter'], 'context': context})

        elif isinstance(item, DoclingTextItem) and item_id not in consumed_text_ids and item.label != 'heading':
            adapter = DoclingTextAdapter(item, doc)
            element_text = adapter.text.strip()
            if len(element_text) > 25 and not any(pattern.search(element_text) for pattern in junk_filter_patterns):
                context = Context(pdf_file_path.stem, page_no, section_title, "Text", "")
                text_items_to_process.append({'adapter': adapter, 'context': context})

    if table_items_to_process:
        table_texts = [item['adapter'].text for item in table_items_to_process]
        table_keywords_list = batch_extract_nlp_keywords(table_texts, nlp, tokenizer,
            max_keywords_per_item=max_keywords + 2)
        for i, item in enumerate(table_items_to_process):
            item['context'].relevant_keywords = table_keywords_list[i]

    if text_items_to_process:
        text_texts = [item['adapter'].text for item in text_items_to_process]
        text_keywords_list = batch_extract_nlp_keywords(text_texts, nlp, tokenizer, max_keywords_per_item=max_keywords)
        for i, item in enumerate(text_items_to_process):
            item['context'].relevant_keywords = text_keywords_list[i]

    final_table_elements = [item['adapter'] for item in table_items_to_process]
    final_table_contexts = [item['context'] for item in table_items_to_process]

    final_text_elements = [item['adapter'] for item in text_items_to_process]
    final_text_contexts = [item['context'] for item in text_items_to_process]

    print(
        f"--- Finished processing. Found {len(final_table_elements)} tables and {len(final_text_elements)} text elements. ---")
    return (final_table_elements, final_table_contexts), (final_text_elements, final_text_contexts)

def _chunk_text_by_section(
    section_texts: Dict[str, List[Tuple[str, Any]]],
    tokenizer: Encoding,
    pdf_file_path: pathlib.Path,
    company_ticker: str
) -> List[Document]:
    """
    Chunks text elements within each section based on a token limit.

    This function iterates through text elements grouped by section, aggregates them
    into chunks that respect MAX_CHUNK_TOKENS, and formats them into Document objects
    with appropriate metadata.

    Args:
        section_texts: A dictionary mapping section titles to lists of (text, context) tuples.
        tokenizer: The tokenizer used to count tokens.
        pdf_file_path: The path to the source PDF file.
        company_ticker: The company ticker symbol.

    Returns:
        A list of chunked Document objects for all text sections.
    """
    final_text_chunks: List[Document] = []

    for section_title, content_list in section_texts.items():
        current_chunk_texts = []
        current_chunk_contexts = []
        current_token_count = 0

        for text, context in content_list:
            cleaned_text = clean_element_text(text)
            if not cleaned_text:
                continue

            element_tokens = len(tokenizer.encode(cleaned_text))

            # If adding the new element exceeds the token limit, finalize the current chunk
            if current_token_count + element_tokens > MAX_CHUNK_TOKENS and current_chunk_texts:
                final_text = "\n\n".join(current_chunk_texts)
                first_context = current_chunk_contexts[0]

                # Aggregate and filter keywords from all contexts in the chunk
                aggregated_candidates = {kw for ctx in current_chunk_contexts for kw in ctx.relevant_keywords}
                final_keywords = _filter_and_clean_keywords(aggregated_candidates, 5, tokenizer)
                first_context.relevant_keywords = final_keywords

                # Create the final augmented content and Document
                context_string = first_context.to_string()
                augmented_content = f"{context_string}\n\n[CONTENT]\n{final_text}"
                final_text_chunks.append(Document(page_content=augmented_content,
                                                  metadata={"source": pdf_file_path.name, "page": first_context.page_number,
                                                            "company": company_ticker, "element_type": "Text",
                                                            "section": section_title,
                                                            "keywords": ", ".join(final_keywords)}))

                # Reset for the next chunk, starting it with the current element
                current_chunk_texts = [cleaned_text]
                current_chunk_contexts = [context]
                current_token_count = element_tokens
            else:
                # Otherwise, add the element to the current chunk
                current_chunk_texts.append(cleaned_text)
                current_chunk_contexts.append(context)
                current_token_count += element_tokens

        # After the loop, process any remaining elements in the last chunk
        if current_chunk_texts:
            final_text = "\n\n".join(current_chunk_texts)
            first_context = current_chunk_contexts[0]

            aggregated_candidates = {kw for ctx in current_chunk_contexts for kw in ctx.relevant_keywords}
            final_keywords = _filter_and_clean_keywords(aggregated_candidates, 5, tokenizer)
            first_context.relevant_keywords = final_keywords

            context_string = first_context.to_string()
            augmented_content = f"{context_string}\n\n[CONTENT]\n{final_text}"
            final_text_chunks.append(Document(page_content=augmented_content,
                                              metadata={"source": pdf_file_path.name, "page": first_context.page_number,
                                                        "company": company_ticker, "element_type": "Text",
                                                        "section": section_title, "keywords": ", ".join(final_keywords)}))

    return final_text_chunks


def load_pdf(pdf_file_path: pathlib.Path, llm: ChatOpenAI, nlp: Language = None, tokenizer: Encoding = None,
             **kwargs) -> List[Document]:
    """
    Loads a PDF, partitions it, and converts elements into intelligently chunked Document objects.
    Tables are returned as separate documents, and text is chunked using a helper function.
    """
    (table_elements, table_contexts), (text_elements, text_contexts) = partition_and_separate_elements(pdf_file_path,
        tokenizer=tokenizer, nlp=nlp, llm=llm, max_keywords=5, **kwargs)

    # If the initial pass finds no tables, rerun the partitioning once in high-resolution mode
    if not table_elements:
        print("No tables detected on initial pass. Retrying in high-resolution mode...")
        (table_elements, table_contexts), (text_elements, text_contexts) = partition_and_separate_elements(
            pdf_file_path, tokenizer=tokenizer, nlp=nlp, llm=llm, max_keywords=5, use_high_res=True)

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

    print(f"Grouping {len(text_elements)} text elements by section...")
    section_texts = defaultdict(list)
    for text_element, context in zip(text_elements, text_contexts):
        section_texts[context.section_title].append((text_element.text, context))

    # The entire text chunking loop is now replaced by a single call
    # to the dedicated helper function.
    print("Calling dedicated function to chunk text elements...")
    text_chunks = _chunk_text_by_section(section_texts, tokenizer, pdf_file_path, company_ticker)
    final_chunks.extend(text_chunks)

    print(f"Total chunks created: {len(final_chunks)}")
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
    master_prompt_template = """You are an Expert Financial Analyst tasked with creating rich, single-sentence summaries of financial tables for a state-of-the-art Retrieval-Augmented Generation (RAG) system. Your entire response must be a single, clean JSON object.

**JSON OUTPUT SPECIFICATION:**
- You will output a single JSON object.
- The object must contain one key: `"summary"`.
- The value must be a single, comprehensive summary sentence.

**RULES FOR THE SUMMARY SENTENCE:**
1.  **The Analyst's Perspective:** Your primary goal is to describe what the table *allows an analyst to do*. Use conceptual language like "analyze trends," "compare performance," "assess financial health," or "understand the breakdown of...".
2.  **Keyword-in-Sentence:** Weave the table's most important row and column headers directly into the narrative of the sentence. This embeds keywords naturally.
3.  **Combine Specifics with Concepts:** Your sentence must merge specific entities from the table (e.g., `iPhone`, `Total Assets`, `2023`) with broader financial concepts (e.g., `revenue streams`, `financial position`, `year-over-year`).
4.  **No Numerical Data:** Do not include any specific numbers or dollar amounts from the table cells.
5.  **Be Concise but Dense:** The sentence should be a single, flowing thought, but packed with as much descriptive, searchable context as possible.
6. **Identify the Core Metric:** If the table's primary metric (e.g., 'Inventory,' 'Employees') is not an explicit header, infer it from the context and state it clearly at the beginning of the summary.

---
**Example 1:**
---
[USER]
Table:
| Category | 2023 | 2022 |
| :--- | :--- | :--- |
| iPhone | $200,583 | $205,489 |
| Mac | $29,357 | $40,177 |
| Services | $85,200 | $78,129 |

[ASSISTANT]
{{
  "summary": "A financial summary comparing the operating performance and net sales revenue streams of key product categories, specifically detailing the results for iPhone, Mac, and Services across the fiscal years 2023 and 2022."
}}
---

**Example 2:**
---
[USER]
Table:
| | 2023 | 2022 |
| :--- | :--- | :--- |
| Total assets | 352,583 | 352,755 |
| Total liabilities | 290,437 | 302,083 |

[ASSISTANT]
{{
  "summary": "A consolidated balance sheet for assessing the company's financial position, providing a year-over-year comparison of its Total Assets against its Total Liabilities for the fiscal periods ending in 2023 and 2022."
}}
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


def count_tokens(text: str, encoder: Encoding) -> int:
    """Counts the number of tokens in a string using the provided tiktoken encoder."""
    return len(encoder.encode(text, disallowed_special=()))


def _filter_and_clean_keywords(candidates: Set[str], max_keywords: int, encoder: Encoding) -> List[str]:
    """Internal helper to filter, clean, and deduplicate keyword candidates."""
    final_keywords, seen_lower = [], set()

    sorted_candidates = sorted([str(c) for c in candidates], key=len, reverse=True)

    for keyword in sorted_candidates:
        kw_clean = keyword.strip(" “”)’'.,:()").replace('’', "'").strip()
        kw_lower = kw_clean.lower()

        if (not kw_clean or len(kw_clean) < 4 or kw_lower in STOP_WORDS or kw_lower.__contains__("thereunto") or any(
                char.isdigit() for char in kw_clean) or count_tokens(kw_clean, encoder) > 7):
            continue

        is_redundant_substring = any(kw_lower in seen for seen in seen_lower)
        if not is_redundant_substring:
            final_keywords.append(kw_clean)
            seen_lower.add(kw_lower)

        if len(final_keywords) >= max_keywords:
            break
    return final_keywords


def batch_extract_nlp_keywords(items_for_processing, nlp_model: Language, encoder_model: Encoding,
                               max_keywords_per_item: int = 5) -> List[List[str]]:
    """
    Extracts keywords from a BATCH of texts using preloaded spaCy and tiktoken models.
    """
    all_results = []
    cleaned_texts = [re.sub(r'^\s*([a-zA-Z0-9]+\.|-|\*)\s+', '', item, flags=re.MULTILINE) for item in
                     items_for_processing]
    cleaned_texts = [' '.join(text.split()) for text in cleaned_texts]

    docs = nlp_model.pipe(cleaned_texts, batch_size=500)

    for doc in docs:
        candidates = set()
        for chunk in doc.noun_chunks:
            candidates.add(chunk.text)
        for ent in doc.ents:
            candidates.add(ent.text)

        keywords = _filter_and_clean_keywords(candidates, max_keywords_per_item, encoder=encoder_model)
        all_results.append(keywords)

    return all_results


# ==============================================================================
#  Main Execution Block - For Debugging and Testing
# ==============================================================================
def main():
    """
    Main function to execute the PDF loading and processing pipeline.
    """
    pdf_to_process = pathlib.Path("./../../../../reports/added/pltr-20231231.pdf")

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
        print(f"Page:     {doc.metadata.get('page', 'N/A')}")
        print(f"Section:  {doc.metadata.get('section', 'N/A')}")
        print(f"Keywords: {doc.metadata.get('keywords', 'N/A')}")
        print("\n--- Content ---\n")
        print(doc.page_content)
        print("\n-----------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
