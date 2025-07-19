# Standard Library Imports
import os
import pathlib
import re
import tempfile
from typing import Dict, List, Set, Tuple, Any

# Docling Imports
from docling.datamodel.layout_model_specs import (DOCLING_LAYOUT_EGRET_LARGE, DOCLING_LAYOUT_EGRET_XLARGE)
from docling.datamodel.pipeline_options import LayoutOptions, EasyOcrOptions
from docling.datamodel.pipeline_options import (TableFormerMode, TableStructureOptions)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


# Langchain Imports
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Tokenizers & NLP Tools
from tiktoken import get_encoding
from tiktoken.core import Encoding
from spacy import load
from spacy.language import Language

# Local Imports
from finquery_parser.types import *
from finquery_app.config import LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, LMSTUDIO_MODEL_NAME

# ==============================================================================
#  Constants
# ==============================================================================
MAX_CHUNK_TOKENS = 256
MIN_CHUNK_TOKENS = 175

MARKDOWN_CLEANING_PATTERNS = [
    # Find all '## table of contents' (as they are artifacts usually)
    re.compile(r"^## table of contents\s*$", re.IGNORECASE),
    # Find page number artifacts
    re.compile(r"^\s*(\|?\s*page\s*\|?\s*)?\d+(\s*of\s*\d+)?\s*\|?\s*$", re.IGNORECASE),
    # Find list item page anchor artifacts
    re.compile(r"^([*\-]) \[PAGE \d+]\(#\d+\)\s*$"),
    # Find the FORM 10-K artifacts
    re.compile(r".*Form 10-K\s*\|\s*\d+\s*$", re.IGNORECASE | re.MULTILINE)]

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

def _create_and_clean_markdown_from_pdf(pdf_file_path: pathlib.Path, temp_dir: str, use_high_res: bool = False, is_tricky_document: bool = False) -> str:
    print("Converting PDF to .md...")

    raw_md_path = os.path.join(temp_dir, f"{pdf_file_path.stem}.md")

    try:
        accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CPU
        )

        if is_tricky_document:
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE,
                    do_cell_matching=False
                ),
                layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_XLARGE),
                images_scale=2.0,
                generate_page_images=True,
                accelerator_options=AcceleratorOptions(
                    num_threads=8, device=AcceleratorDevice.CPU
                ),
                ocr_options=EasyOcrOptions(
                    lang=["en"],
                    confidence_threshold=0.4,
                    force_full_page_ocr=True
                )
            )
        elif use_high_res:
            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE,
                    do_cell_matching=False
                ),
                layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_LARGE),
                images_scale=2.0,
                accelerator_options=accelerator_options
            )
        else:
            pipeline_options = PdfPipelineOptions(
                do_table_structure=True,
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE, do_cell_matching=False),
                accelerator_options=accelerator_options
            )

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

        result = converter.convert(pdf_file_path)

        if not result or not result.document:
            return ""

        markdown_content = result.document.export_to_markdown()

        with open(raw_md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"Converted PDF and saved raw markdown to '{raw_md_path}'")

    except Exception as e:
        print(f"PDF to .md conversion failed for {pdf_file_path}: {e}")
        return ""

    cleaned_md_path = os.path.join(temp_dir, f"{pdf_file_path.stem}_cleaned.md")

    lines_removed = 0

    with open(raw_md_path, 'r', encoding='utf-8') as infile, open(cleaned_md_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not any(pattern.search(line) for pattern in MARKDOWN_CLEANING_PATTERNS):
                outfile.write(line)
            else:
                lines_removed += 1

    return cleaned_md_path


def _parse_markdown_for_elements(md_path: str) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reads cleaned markdown, extracts tables (with their prefaces), and groups remaining text.
    This version uses a simpler, more direct method for preface detection.
    """
    print("Step 3: Parsing Markdown for tables and text sections...")
    tables, texts, current_headers, is_in_table, current_block = [], [], [], False, []

    MAX_PREFACE_LENGTH = 600
    PREFACE_KEYWORDS = ['the following table', 'consisted of the following', 'as follows:', 'were as follows',
                        'summarizes', 'presents', 'sets forth', '(in thousands', '(in millions', 'except per share']

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def finalize_block():
        nonlocal current_block, is_in_table, texts, tables, current_headers
        if not current_block: return

        block_content = "".join(current_block).strip()
        if not block_content:
            current_block = []
            return

        if is_in_table:
            table_element = {'headers': list(current_headers), 'content': block_content}

            if texts and len(texts[-1]['content']) < MAX_PREFACE_LENGTH:
                last_text_content = texts[-1]['content'].lower()
                if any(keyword in last_text_content for keyword in PREFACE_KEYWORDS):
                    # It's a preface! Pop it from the texts list and attach it to the table.
                    preface_element = texts.pop()
                    table_element['preface'] = preface_element['content']

            tables.append(table_element)
        else:
            paragraphs = re.split(r'\n\s*\n', block_content)
            for para in paragraphs:
                if para.strip():
                    texts.append({'headers': list(current_headers), 'content': para.strip()})

        current_block, is_in_table = [], False

    for line in lines:
        match = re.match(r"^(#+)\s(.*)", line)
        if match:
            finalize_block()
            level, title = len(match.group(1)), match.group(2).strip()
            current_headers = [h for h in current_headers if h['level'] < level]
            current_headers.append({'level': level, 'title': title})
            continue

        is_table_line = re.match(r"^\s*\|.*\|", line)
        if is_table_line and not is_in_table:
            finalize_block()
            is_in_table = True
        elif not is_table_line and is_in_table:
            finalize_block()
        current_block.append(line)

    finalize_block()

    print(f"Found {len(tables)} tables and {len(texts)} text blocks (after associating prefaces).")
    return tables, texts


def _chunk_text_elements(text_elements, tokenizer, source_pdf_path, company_ticker, nlp) -> List[Document]:
    """
    Performs content-aware chunking and merges excessively short chunks.
    """
    print(f"Chunking {len(text_elements)} text blocks...")
    if not text_elements: return []

    # Run keyword extraction on all text elements for high batch efficiency
    all_keywords = batch_extract_nlp_keywords([e['content'] for e in text_elements], nlp, tokenizer, 5)
    for i, elem in enumerate(text_elements): elem['keywords'] = all_keywords[i]

    intermediate_chunks = []
    current_chunk_texts, current_chunk_keywords, current_token_count, current_section_title = [], set(), 0, "Introduction"

    def finalize_initial_chunk():
        nonlocal current_chunk_texts, current_chunk_keywords, current_token_count, current_section_title

        if not current_chunk_texts:
            return

        intermediate_chunks.append({
            "texts": current_chunk_texts,
            "keywords": current_chunk_keywords,
            "section": current_section_title
        })

        current_chunk_texts, current_chunk_keywords, current_token_count = [], set(), 0

    for element in text_elements:
        cleaned_text = clean_element_text(element['content'])

        if not cleaned_text:
            continue

        element_section = element['headers'][-1]['title'] if element['headers'] else "Introduction"
        element_tokens = len(tokenizer.encode(cleaned_text))

        if (current_token_count + element_tokens > MAX_CHUNK_TOKENS and current_chunk_texts) or \
                (element_section != current_section_title and current_chunk_texts):
            finalize_initial_chunk()

        current_chunk_texts.append(cleaned_text)
        current_chunk_keywords.update(element['keywords'])
        current_token_count += element_tokens
        current_section_title = element_section

    finalize_initial_chunk()

    if not intermediate_chunks: return []

    merged_chunks = []
    current_merged_chunk = intermediate_chunks[0]

    for i in range(1, len(intermediate_chunks)):
        next_chunk = intermediate_chunks[i]

        next_chunk_text = "\n\n".join(next_chunk['texts'])
        next_chunk_tokens = count_tokens(next_chunk_text, tokenizer)

        if next_chunk_tokens < MIN_CHUNK_TOKENS and next_chunk['section'] == current_merged_chunk['section']:
            current_merged_chunk['texts'].extend(next_chunk['texts'])
            current_merged_chunk['keywords'].update(next_chunk['keywords'])
        else:
            merged_chunks.append(current_merged_chunk)
            current_merged_chunk = next_chunk

    merged_chunks.append(current_merged_chunk)

    final_text_chunks = []
    for chunk in merged_chunks:
        final_text = "\n\n".join(chunk['texts'])
        final_keywords = _filter_and_clean_keywords(chunk['keywords'], 5, tokenizer)
        section = chunk['section']

        context = Context(source_pdf_path.stem, 1, section, "Text", summary="")
        context.relevant_keywords = final_keywords

        augmented_content = f"{context.to_string()}\n\n[CONTENT]\n{final_text}"

        final_text_chunks.append(Document(page_content=augmented_content, metadata={
            "source": source_pdf_path.name, "company": company_ticker,
            "element_type": "Text", "section": section, "keywords": ", ".join(final_keywords)
        }))

    return final_text_chunks


def load_pdf(pdf_file_path: pathlib.Path, llm: ChatOpenAI, nlp: Language, tokenizer: Encoding, use_high_res: bool, filter_small_elements: bool = True) -> \
List[Document]:
    """Main orchestrator: Loads a PDF, converts, cleans, parses, and chunks it."""
    company_ticker = pdf_file_path.stem.split('-')[0].upper()
    final_chunks = []
    parser = define_parser()

    with tempfile.TemporaryDirectory() as temp_dir:
        cleaned_md_path_str = _create_and_clean_markdown_from_pdf(pdf_file_path, temp_dir)

        if not cleaned_md_path_str:
            return []

        tables, text_elements = _parse_markdown_for_elements(cleaned_md_path_str)

        if not tables:
            _create_and_clean_markdown_from_pdf(pdf_file_path, temp_dir, use_high_res)

    if tables:
        print(f"Processing {len(tables)} table chunks...")

        table_contents_for_keywords = [
            f"{table.get('preface', '')}\n\n{table['content']}" for table in tables
        ]

        table_keywords_list = batch_extract_nlp_keywords(table_contents_for_keywords, nlp, tokenizer, 7)

        for i, table in enumerate(tables):
            section_title = table['headers'][-1]['title'] if table['headers'] else "Financial Table"
            preface = table.get('preface', '')

            full_content = f"{preface}\n\n{table['content']}".strip()

            try:
                summary_result = get_one_line_summary(full_content, section_title, parser, llm)
                summary_text = summary_result.get('summary', "Error: Summary key missing.")
            except Exception as e:
                print(f"Summarization failed for table in section '{section_title}': {e}")
                summary_text = "Error: Table summarization failed."

            context = Context(pdf_file_path.stem, 1, section_title, "Table", summary=summary_text, table_prefix=preface)
            context.relevant_keywords = table_keywords_list[i]

            augmented_content = f"{context.to_string()}\n\n[CONTENT]\n{full_content}"

            final_chunks.append(Document(page_content=augmented_content, metadata={
                "source": pdf_file_path.name, "company": company_ticker,
                "element_type": "Table", "section": section_title, "keywords": ", ".join(context.relevant_keywords)}))

    if text_elements:
        text_chunks = _chunk_text_elements(text_elements, tokenizer, pdf_file_path, company_ticker, nlp)
        final_chunks.extend(text_chunks)

    print(f"\nTotal chunks created: {len(final_chunks)}")

    if filter_small_elements:
        final_chunks = [doc for doc in final_chunks if len(doc.page_content.split("[CONTENT]\n", 1)[1]) >= 200]
        print(f"Filtered out small chunks. Returning {len(final_chunks)} final chunks.")

    return final_chunks


# ==============================================================================
#  Helper and Utility Functions
# ==============================================================================
def clean_element_text(text: str, **kwargs): return text.strip()


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
        kw_clean = keyword.strip(" “”)’'.,:()|").replace('’', "'").strip()
        kw_lower = kw_clean.lower()

        if (not kw_clean or len(kw_clean) < 4 or kw_lower in STOP_WORDS or kw_lower.__contains__("thereunto") or any(
                char.isdigit() for char in kw_clean) or count_tokens(kw_clean, encoder) > 7):
            continue

        is_redundant_substring = any(kw_lower in seen for seen in seen_lower)

        if not is_redundant_substring:
            # A small fix to remove unneeded prepositions
            final_keywords.append(re.sub(r'^(?:A|An|The|Their)\s+', '', kw_clean, flags=re.IGNORECASE))
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


def get_document_summary(chunks: List[str]) -> str:
    # get all headings and table summaries from content
    # if there are <10 chunks then just return the parsed text
    return


# ==============================================================================
#  Main Execution Block
# ==============================================================================
def main():
    """
    Main function to execute the PDF loading and processing pipeline.
    """
    pdf_to_process = pathlib.Path("./../../../../../../../various_sample_files_intl_and_us_various_types_and_formats/indian_financial_statement_societe_generale.pdf")

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

    documents = load_pdf(pdf_to_process, llm=llm, use_high_res=True, nlp=nlp, tokenizer=tiktoken_encoding)

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
