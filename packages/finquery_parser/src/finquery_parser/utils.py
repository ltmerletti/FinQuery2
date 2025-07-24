# Standard Library Imports
import json
import pathlib
import re
import tempfile
from typing import Set, Tuple, List, Dict, Any, Optional

# Docling Imports
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import (DOCLING_LAYOUT_EGRET_LARGE, DOCLING_LAYOUT_EGRET_XLARGE)
from docling.datamodel.pipeline_options import (LayoutOptions, EasyOcrOptions, PictureDescriptionVlmOptions,
                                                PdfPipelineOptions, TableFormerMode, TableStructureOptions)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Langchain Imports
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Tokenizers & NLP Tools
from spacy.language import Language
from tiktoken.core import Encoding

# Local Imports
from finquery_app.config import SUMMARY_DIRECTORY
from finquery_parser.config import VISION_MODEL_ID, MARKDOWN_CLEANING_PATTERNS, PREFACE_KEYWORDS, MAX_PREFACE_LENGTH, \
    COVER_PAGE_TOKEN_COUNT, TIER_1_SNIPPET_TOKEN_LIMIT, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, STOP_WORDS, \
    TABLE_SUMMARY_PROMPT, DOCUMENT_SUMMARY_PROMPT, METADATA_EXTRACTION_PROMPT, DOCUMENT_TYPE_AND_SCHEMA_PROMPT
from finquery_parser.types import Context, TableSummary, DatabaseInterface, PDFConversionError, TableElement, \
    ProseElement, Header, MetadataExtractionError, TableSummarizationError, IntermediateChunk


# ==============================================================================
#  Main Orchestration Function
# ==============================================================================

def load_pdf(pdf_file_path: pathlib.Path, llm: ChatOpenAI, nlp: Language, tokenizer: Encoding,
             db_interface: Optional[DatabaseInterface] = None, use_high_res: bool = False,
             filter_small_elements: bool = True, small_llm: Optional[ChatOpenAI] = None) -> List[Document]:
    """
    Main orchestrator to load, parse, enrich, and chunk a PDF document.

    Args:
        pdf_file_path: Path to the input PDF file.
        llm: The primary language model for content processing.
        nlp: The spaCy language model for NLP tasks.
        tokenizer: The tiktoken tokenizer for counting tokens.
        db_interface: An optional database interface for metadata operations.
        use_high_res: Flag to use higher-resolution PDF parsing.
        filter_small_elements: Flag to filter out very short chunks.
        small_llm: An optional, smaller language model for faster tasks.

    Returns:
        A list of LangChain Document objects ready for ingestion.
    """
    company_ticker = pdf_file_path.stem.split('-')[0].upper()

    print("\n--- Step 1: Parsing Document ---")
    try:
        cleaned_md_content, table_elements, prose_elements = _parse_document(pdf_file_path, use_high_res)
    except (PDFConversionError, FileNotFoundError) as e:
        print(f"ERROR: Document parsing failed for {pdf_file_path}. Reason: {e}")
        return []

    if not cleaned_md_content:
        return []

    print("\n--- Step 2: Running Intelligence Layer ---")
    extracted_metadata, needs_human_review = _run_intelligence_layer(
        llm, small_llm, db_interface, tokenizer, cleaned_md_content, prose_elements, table_elements
    )
    if needs_human_review:
        print("\n--> This document has been flagged for HUMAN REVIEW.")

    print("\n--- Step 3: Enriching and Chunking Document Elements ---")
    final_chunks, table_contexts_for_summary = _enrich_and_chunk_elements(
        prose_elements, table_elements, extracted_metadata, pdf_file_path, company_ticker, nlp, tokenizer, llm
    )

    _generate_final_summary(cleaned_md_content, table_contexts_for_summary, pdf_file_path, small_llm or llm)

    print(f"\nTotal chunks created: {len(final_chunks)}")

    if filter_small_elements:
        original_count = len(final_chunks)
        final_chunks = [doc for doc in final_chunks if len(doc.page_content.split("[CONTENT]\n", 1)[1]) >= 200]
        print(f"Filtered {original_count - len(final_chunks)} small chunks. Returning {len(final_chunks)} final chunks.")

    return final_chunks


# ==============================================================================
#  Stage 1: Document Parsing and Cleaning
# ==============================================================================

def _parse_document(pdf_file_path: pathlib.Path, use_high_res: bool) -> Tuple[str, List[TableElement], List[ProseElement]]:
    """
    Handles PDF-to-Markdown conversion, cleaning, and element extraction.

    Args:
        pdf_file_path: The path to the PDF file.
        use_high_res: A flag indicating whether to use high-resolution parsing.

    Returns:
        A tuple containing the full cleaned markdown content, a list of table
        elements, and a list of prose elements.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        try:
            cleaned_md_path = _convert_and_clean_pdf(pdf_file_path, temp_dir_path)
            with open(cleaned_md_path, 'r', encoding='utf-8') as f:
                cleaned_md_content = f.read()
            table_elements, prose_elements = _parse_markdown_for_elements(cleaned_md_path)

            # If parsing fails, retry with higher resolution settings
            if not table_elements and not prose_elements:
                print("Initial parsing yielded no content. Retrying with higher resolution...")
                cleaned_md_path = _convert_and_clean_pdf(pdf_file_path, temp_dir_path, use_high_res=True)
                with open(cleaned_md_path, 'r', encoding='utf-8') as f:
                    cleaned_md_content = f.read()
                table_elements, prose_elements = _parse_markdown_for_elements(cleaned_md_path)

        except (PDFConversionError, FileNotFoundError) as e:
            raise PDFConversionError(f"Failed during document parsing stage: {e}") from e

    return cleaned_md_content, table_elements, prose_elements


def _convert_and_clean_pdf(
    pdf_file_path: pathlib.Path,
    temp_dir_path: pathlib.Path,
    use_high_res: bool = False,
    is_tricky_document: bool = False,
    describe_images: bool = False
) -> pathlib.Path:
    """
    Converts a PDF to a raw markdown file and then cleans it.

    Args:
        pdf_file_path: Path to the source PDF.
        temp_dir_path: Path to the temporary directory for output files.
        use_high_res: Flag for high-resolution layout model.
        is_tricky_document: Flag for documents requiring full OCR.
        describe_images: Flag to enable image descriptions.

    Returns:
        The path to the cleaned markdown file.
    """
    print(f"Converting '{pdf_file_path.name}' to Markdown...")
    raw_md_path = _convert_pdf_to_raw_markdown(
        pdf_file_path, temp_dir_path, use_high_res, is_tricky_document, describe_images
    )
    print(f"Cleaning raw markdown file: '{raw_md_path.name}'")
    cleaned_md_path = _clean_raw_markdown(raw_md_path, temp_dir_path)
    return cleaned_md_path


def _get_pdf_pipeline_options(
    use_high_res: bool, is_tricky_document: bool, describe_images: bool
) -> PdfPipelineOptions:
    """Constructs the appropriate Docling pipeline options based on flags."""
    accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CPU)

    if is_tricky_document:
        return PdfPipelineOptions(
            do_ocr=True, do_table_structure=True,
            table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE, do_cell_matching=False),
            layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_XLARGE), images_scale=2.0,
            generate_page_images=True,
            accelerator_options=AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CPU),
            ocr_options=EasyOcrOptions(lang=["en"], confidence_threshold=0.4, force_full_page_ocr=True)
        )

    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE, do_cell_matching=False),
        accelerator_options=accelerator_options
    )

    if use_high_res:
        pipeline_options.layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_LARGE)
        pipeline_options.images_scale = 2.0

    if describe_images:
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
            repo_id=VISION_MODEL_ID,
            prompt="Describe the image in three sentences. Be concise and accurate."
        )
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_picture_images = True

    return pipeline_options


def _convert_pdf_to_raw_markdown(
    pdf_file_path: pathlib.Path,
    temp_dir_path: pathlib.Path,
    use_high_res: bool,
    is_tricky_document: bool,
    describe_images: bool
) -> pathlib.Path:
    """Uses Docling to convert a PDF file to a raw markdown file."""
    raw_md_path = temp_dir_path / f"{pdf_file_path.stem}.md"

    try:
        pipeline_options = _get_pdf_pipeline_options(use_high_res, is_tricky_document, describe_images)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        result = converter.convert(pdf_file_path)

        if not result or not result.document:
            raise PDFConversionError("Docling converter returned no result or document.")

        markdown_content = result.document.export_to_markdown()
        raw_md_path.write_text(markdown_content, encoding="utf-8")
        print(f"Saved raw markdown to '{raw_md_path}'")
        return raw_md_path

    except Exception as e:
        raise PDFConversionError(f"PDF to .md conversion failed for {pdf_file_path.name}: {e}") from e


def _clean_raw_markdown(raw_md_path: pathlib.Path, temp_dir_path: pathlib.Path) -> pathlib.Path:
    """Reads a raw markdown file and writes a cleaned version by removing artifact lines."""
    cleaned_md_path = temp_dir_path / f"{raw_md_path.stem}_cleaned.md"
    lines_removed = 0

    with open(raw_md_path, 'r', encoding='utf-8') as infile, open(cleaned_md_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not any(pattern.search(line) for pattern in MARKDOWN_CLEANING_PATTERNS):
                outfile.write(line)
            else:
                lines_removed += 1

    print(f"Removed {lines_removed} artifact lines.")
    return cleaned_md_path


def _parse_markdown_for_elements(md_path: pathlib.Path) -> Tuple[List[TableElement], List[ProseElement]]:
    """
    Reads cleaned markdown and parses it into structured table and prose elements.

    This function uses a state machine to iterate through the lines of the markdown
    file, grouping consecutive text lines into prose elements and table-formatted
    lines into table elements. It also associates potential "preface" text with
    the tables that follow them.

    Args:
        md_path: The path to the cleaned markdown file.

    Returns:
        A tuple containing a list of parsed table elements and a list of prose elements.
    """
    print("Parsing Markdown for tables and text sections...")
    tables: List[TableElement] = []
    texts: List[ProseElement] = []
    current_headers: List[Header] = []
    is_in_table = False
    current_block: List[str] = []
    element_order = 0

    lines = md_path.read_text(encoding='utf-8').splitlines(True)

    def finalize_block():
        nonlocal current_block, is_in_table, element_order
        if not current_block:
            return

        block_content = "".join(current_block).strip()
        if not block_content:
            current_block = []
            return

        if is_in_table:
            table_element: TableElement = {'headers': list(current_headers), 'content': block_content, 'order': element_order}
            # Heuristically check if the previous text block is a preface
            if texts and len(texts[-1]['content']) < MAX_PREFACE_LENGTH:
                last_text_content = texts[-1]['content'].lower()
                if any(keyword in last_text_content for keyword in PREFACE_KEYWORDS):
                    preface_element = texts.pop()
                    table_element['preface'] = preface_element['content']
            tables.append(table_element)
        else:
            paragraphs = re.split(r'\n\s*\n', block_content)
            for para in paragraphs:
                if para.strip():
                    # Placeholder for keywords; will be populated later
                    texts.append({'headers': list(current_headers), 'content': para.strip(), 'order': element_order, 'keywords': []})
                    element_order += 1

        current_block, is_in_table = [], False
        element_order += 1

    for line in lines:
        header_match = re.match(r"^(#+)\s(.*)", line)
        if header_match:
            finalize_block()
            level, title = len(header_match.group(1)), header_match.group(2).strip()
            # Pop headers of the same or greater level
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
    print(f"Found {len(tables)} tables and {len(texts)} text blocks.")
    return tables, texts


# ==============================================================================
#  Stage 2: Intelligence Layer (Metadata Extraction)
# ==============================================================================

def _run_intelligence_layer(
    llm: ChatOpenAI, small_llm: Optional[ChatOpenAI], db_interface: Optional[DatabaseInterface], tokenizer: Encoding,
    cleaned_md_content: str, prose_elements: List[ProseElement], table_elements: List[TableElement]
) -> Tuple[Dict[str, Any], bool]:
    """
    Orchestrates the decider and extraction agents to produce structured metadata.

    Args:
        llm: The primary (powerful) language model for extraction.
        small_llm: A smaller, faster language model for classification.
        db_interface: The database interface to check for known document types.
        tokenizer: The tokenizer for text manipulation.
        cleaned_md_content: The full markdown content of the document.
        prose_elements: Parsed prose elements.
        table_elements: Parsed table elements.

    Returns:
        A tuple containing the extracted metadata dictionary and a boolean
        indicating if human review is needed.
    """
    if not (small_llm and db_interface):
        print("Skipping intelligence layer: small_llm or db_interface not provided.")
        return {}, False

    all_elements = sorted(prose_elements + table_elements, key=lambda x: x.get('order', 0))
    document_outline = "\n".join(sorted({f"{'#' * h['level']} {h['title']}" for e in all_elements for h in e.get('headers', [])}))
    known_doc_types = db_interface.get_known_doc_types()

    # Decide on document type and get the schema
    doc_type_decision = _decide_document_type_and_schema(small_llm, tokenizer, cleaned_md_content, document_outline, known_doc_types)
    metadata_schema_to_use, doc_type_name = _get_metadata_schema_from_decision(doc_type_decision, db_interface, known_doc_types)

    if metadata_schema_to_use:
        try:
            extracted_metadata, needs_human_review = _run_extraction_agent(llm, prose_elements, table_elements, metadata_schema_to_use, tokenizer)
            extracted_metadata['document_type'] = doc_type_name
            return extracted_metadata, needs_human_review
        except MetadataExtractionError as e:
            print(f"ERROR: {e}")
            return {}, True  # Flag for human review on failure
    else:
        print("Could not determine a metadata schema. Skipping extraction.")
        return {}, True  # Flag for human review if no schema is found


def _get_metadata_schema_from_decision(
    decision: Dict[str, Any], db: DatabaseInterface, known_types: List[Dict]
) -> Tuple[Optional[Dict], Optional[str]]:
    """Processes the decision from the decider agent to return a schema and type name."""
    action = decision.get("action")

    if action == "create" and decision.get("new_type_name") and decision.get("new_metadata_schema"):
        new_type = db.create_doc_type(type_name=decision["new_type_name"], schema=decision["new_metadata_schema"])
        return new_type.get("metadata_schema"), new_type.get("type_name")
    elif action == "match" and decision.get("type_id"):
        matched_type = next((t for t in known_types if t.get('id') == decision["type_id"]), None)
        if matched_type:
            return matched_type.get("metadata_schema"), matched_type.get("type_name")

    return None, None


def _decide_document_type_and_schema(
        small_llm: ChatOpenAI, tokenizer: Encoding, full_markdown_content: str, document_outline: str,
        known_doc_types: List[Dict]
) -> Dict[str, Any]:
    """
    Analyzes document context to match or create a document type schema.
    """
    print("\n--- Running Document Type & Schema Decider ---")
    cover_page_text = tokenizer.decode(tokenizer.encode(full_markdown_content)[:COVER_PAGE_TOKEN_COUNT])
    known_doc_types_json = json.dumps(known_doc_types)

    prompt_template = DOCUMENT_TYPE_AND_SCHEMA_PROMPT

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | small_llm | (lambda x: _extract_json_from_string(x.content)) | JsonOutputParser()

    try:
        result = chain.invoke({
            "known_doc_types_json": known_doc_types_json,
            "cover_page_text": cover_page_text,
            "document_outline": document_outline
        })
        print(f"Decider Agent decision: {result}")
        return result
    except (OutputParserException, Exception) as e:
        print(f"Decider Agent failed: {e}")
        return {"action": "error", "details": str(e)}


def _run_extraction_agent(
    llm: ChatOpenAI, prose_elements: List[ProseElement], table_elements: List[TableElement],
    metadata_schema: Dict, tokenizer: Encoding
) -> Tuple[Dict[str, Any], bool]:
    """
    Runs a tiered extraction process to populate a metadata schema.

    Args:
        llm: The powerful language model for extraction.
        prose_elements: A list of prose elements from the document.
        table_elements: A list of table elements from the document.
        metadata_schema: The target JSON schema for extraction.
        tokenizer: The tokenizer for slicing the document snippet.

    Returns:
        A tuple containing the extracted metadata and a boolean flag for human review.
    """
    print("\n--- Running Tiered Extraction Agent ---")
    print("Tier 1: Analyzing initial document snippet for metadata...")
    all_elements = sorted(prose_elements + table_elements, key=lambda x: x.get('order', 0))
    snippet = _get_initial_snippet(all_elements, TIER_1_SNIPPET_TOKEN_LIMIT, tokenizer)

    try:
        extracted_metadata = _extract_metadata_with_llm(llm, metadata_schema, snippet)
    except MetadataExtractionError as e:
        raise MetadataExtractionError(f"Initial LLM extraction failed: {e}") from e

    print("Tier 2: Surgically searching for missing data...")
    for field, value in extracted_metadata.items():
        if value is None:
            tier_2_value = _surgical_table_extractor_no_llm(table_elements, field)
            if tier_2_value is not None:
                extracted_metadata[field] = tier_2_value

    needs_human_review = any(value is None for value in extracted_metadata.values())
    if needs_human_review:
        missing_fields = [field for field, value in extracted_metadata.items() if value is None]
        print(f"  - VALIDATION FAILED: Fields {missing_fields} are still null. Flagging for human review.")
    else:
        print("  - Validation passed. All metadata fields extracted.")

    return extracted_metadata, needs_human_review


def _get_initial_snippet(elements: List[Dict], token_limit: int, tokenizer: Encoding) -> str:
    """Creates a document snippet up to a specified token limit."""
    snippet_content = []
    current_tokens = 0
    for element in elements:
        content = element.get('content', '')
        element_tokens = count_tokens(content, tokenizer)
        if current_tokens + element_tokens > token_limit:
            remaining_tokens = token_limit - current_tokens
            sliced_content = tokenizer.decode(tokenizer.encode(content)[:remaining_tokens])
            snippet_content.append(sliced_content)
            break
        snippet_content.append(content)
        current_tokens += element_tokens
    return "\n\n---\n\n".join(snippet_content)


def _extract_metadata_with_llm(llm: ChatOpenAI, schema: Dict, snippet: str) -> Dict[str, Any]:
    """Invokes an LLM to extract data from a snippet based on a schema."""
    extraction_prompt_template = METADATA_EXTRACTION_PROMPT
    prompt = ChatPromptTemplate.from_template(extraction_prompt_template)
    chain = prompt | llm | (lambda x: strip_thinking_tags(x.content)) | JsonOutputParser()
    try:
        result = chain.invoke({"schema": json.dumps(schema, indent=2), "snippet": snippet})
        print(f"Tier 1 Extraction Complete: {result}")
        return result
    except (OutputParserException, Exception) as e:
        raise MetadataExtractionError("LLM failed to return valid JSON for metadata.") from e


def _surgical_table_extractor_no_llm(table_elements: List[TableElement], field_to_find: str) -> Optional[str]:
    """
    Performs a targeted, non-LLM regex search for a field within markdown tables.

    Args:
        table_elements: The list of parsed tables.
        field_to_find: The snake_case field name to search for.

    Returns:
        The extracted value as a string, or None if not found.
    """
    print(f"Surgically searching tables for '{field_to_find}'...")
    search_terms = [field_to_find.replace('_', ' ').lower(), field_to_find.lower()]
    pattern = re.compile(r"^\s*\|\s*(" + "|".join(re.escape(term) for term in search_terms) + r")\s*\|", re.IGNORECASE)

    for table in table_elements:
        for line in table.get('content', '').splitlines():
            if pattern.search(line):
                columns = [col.strip() for col in line.split('|') if col.strip()]
                if len(columns) > 1:
                    # Return the first column that looks like a value
                    for value in columns[1:]:
                        if any(char.isdigit() for char in value):
                            print(f"    - Found and extracted value: '{value}'")
                            return value
    return None


# ==============================================================================
#  Stage 3: Element Enrichment and Chunking
# ==============================================================================

def _enrich_and_chunk_elements(
    prose_elements: List[ProseElement], table_elements: List[TableElement], extracted_metadata: Dict,
    pdf_file_path: pathlib.Path, company_ticker: str, nlp: Language, tokenizer: Encoding, llm: ChatOpenAI
) -> Tuple[List[Document], List[Context]]:
    """
    Processes table and prose elements to create final, enriched Document chunks.

    Args:
        prose_elements: List of parsed text elements.
        table_elements: List of parsed table elements.
        extracted_metadata: Metadata to be added to each chunk.
        pdf_file_path: Path to the original PDF for source info.
        company_ticker: The company ticker symbol.
        nlp: spaCy model for keyword extraction.
        tokenizer: Tokenizer for various NLP tasks.
        llm: Language model for table summarization.

    Returns:
        A tuple containing the list of final Document objects and a list of
        table Context objects used for the final summary.
    """
    final_chunks = []
    table_contexts_for_summary = []

    # Process Tables
    if table_elements:
        table_chunks, table_contexts_for_summary = _process_table_elements(
            table_elements, extracted_metadata, pdf_file_path, company_ticker, nlp, tokenizer, llm
        )
        final_chunks.extend(table_chunks)

    # Process and Chunk Prose
    if prose_elements:
        text_chunks = _chunk_text_elements(
            prose_elements, tokenizer, pdf_file_path, company_ticker, nlp, extracted_metadata
        )
        final_chunks.extend(text_chunks)

    return final_chunks, table_contexts_for_summary


def _process_table_elements(
    table_elements: List[TableElement], extracted_metadata: Dict, pdf_file_path: pathlib.Path, company_ticker: str,
    nlp: Language, tokenizer: Encoding, llm: ChatOpenAI
) -> Tuple[List[Document], List[Context]]:
    """Generates summaries and Documents for all table elements."""
    print(f"Processing {len(table_elements)} table chunks...")
    table_chunks = []
    table_contexts = []
    parser = JsonOutputParser(pydantic_object=TableSummary)

    # Batch extract keywords for all tables at once for efficiency
    table_contents_for_nlp = [f"{t.get('preface', '')}\n\n{t['content']}" for t in table_elements]
    table_keywords_list = batch_extract_nlp_keywords(table_contents_for_nlp, nlp, tokenizer, 7)

    for i, table in enumerate(table_elements):
        section_title = table['headers'][-1]['title'] if table['headers'] else "Financial Table"
        preface = table.get('preface', '')
        full_content = f"{preface}\n\n{table['content']}".strip()

        try:
            summary_result = get_one_line_summary(full_content, section_title, parser, llm)
            summary_text = summary_result.get('summary', "Error: Summary key missing.")
        except TableSummarizationError as e:
            print(f"Summarization failed for table in section '{section_title}': {e}")
            summary_text = "Error: Table summarization failed."

        context = Context(pdf_file_path.stem, 1, section_title, "Table", summary=summary_text, table_prefix=preface)
        context.relevant_keywords = table_keywords_list[i]
        table_contexts.append(context)

        augmented_content = f"{context.to_string()}\n\n[CONTENT]\n{full_content}"
        chunk_metadata = {
            "source": pdf_file_path.name, "company": company_ticker, "element_type": "Table",
            "section": section_title, "keywords": ", ".join(context.relevant_keywords)
        }
        chunk_metadata.update(extracted_metadata)
        table_chunks.append(Document(page_content=augmented_content, metadata=chunk_metadata))

    return table_chunks, table_contexts


def _chunk_text_elements(
    prose_elements: List[ProseElement], tokenizer: Encoding, pdf_file_path: pathlib.Path,
    company_ticker: str, nlp: Language, extracted_metadata: Dict
) -> List[Document]:
    """
    Performs content-aware chunking on prose elements.

    Args:
        prose_elements: The list of prose elements to chunk.
        tokenizer: The tokenizer for counting tokens.
        pdf_file_path: Path for sourcing metadata.
        company_ticker: Company ticker for metadata.
        nlp: spaCy model for keyword extraction.
        extracted_metadata: Previously extracted metadata.

    Returns:
        A list of LangChain Document objects for the text chunks.
    """
    if not prose_elements:
        return []
    print(f"Chunking {len(prose_elements)} text blocks...")

    # Enrich prose elements with keywords in a batch
    all_keywords = batch_extract_nlp_keywords([e['content'] for e in prose_elements], nlp, tokenizer, 5)
    for i, elem in enumerate(prose_elements):
        elem['keywords'] = all_keywords[i]

    initial_chunks = _create_initial_text_chunks(prose_elements, tokenizer)
    if not initial_chunks:
        return []

    merged_chunks = _merge_small_chunks(initial_chunks, tokenizer)
    return _format_chunks_as_documents(merged_chunks, pdf_file_path, company_ticker, extracted_metadata, tokenizer)


def _create_initial_text_chunks(prose_elements: List[ProseElement], tokenizer: Encoding) -> List[IntermediateChunk]:
    """First pass of chunking: group texts by section and token limits."""
    intermediate_chunks: List[IntermediateChunk] = []
    current_texts, current_keywords, current_tokens, current_section = [], set(), 0, "Introduction"

    for element in prose_elements:
        cleaned_text = element['content'].strip()
        if not cleaned_text:
            continue

        element_section = element['headers'][-1]['title'] if element['headers'] else "Introduction"
        element_tokens = count_tokens(cleaned_text, tokenizer)

        # Finalize chunk if it exceeds token limit or if section changes
        if (current_tokens + element_tokens > MAX_CHUNK_TOKENS and current_texts) or \
           (element_section != current_section and current_texts):
            intermediate_chunks.append({"texts": current_texts, "keywords": current_keywords, "section": current_section})
            current_texts, current_keywords, current_tokens = [], set(), 0

        current_texts.append(cleaned_text)
        current_keywords.update(element['keywords'])
        current_tokens += element_tokens
        current_section = element_section

    # Add the last remaining chunk
    if current_texts:
        intermediate_chunks.append({"texts": current_texts, "keywords": current_keywords, "section": current_section})

    return intermediate_chunks


def _merge_small_chunks(chunks: List[IntermediateChunk], tokenizer: Encoding) -> List[IntermediateChunk]:
    """Second pass of chunking: merge chunks that are too small with their predecessors."""
    if not chunks:
        return []

    merged_chunks: List[IntermediateChunk] = []
    current_merged_chunk = chunks[0]

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        next_chunk_tokens = count_tokens("\n\n".join(next_chunk['texts']), tokenizer)

        # Merge if the next chunk is too small and belongs to the same section
        if next_chunk_tokens < MIN_CHUNK_TOKENS and next_chunk['section'] == current_merged_chunk['section']:
            current_merged_chunk['texts'].extend(next_chunk['texts'])
            current_merged_chunk['keywords'].update(next_chunk['keywords'])
        else:
            merged_chunks.append(current_merged_chunk)
            current_merged_chunk = next_chunk

    merged_chunks.append(current_merged_chunk)
    return merged_chunks


def _format_chunks_as_documents(
    chunks: List[IntermediateChunk], pdf_file_path: pathlib.Path, company_ticker: str,
    extracted_metadata: Dict, tokenizer: Encoding
) -> List[Document]:
    """Converts intermediate chunks into final LangChain Document objects."""
    final_documents = []
    for chunk in chunks:
        final_text = "\n\n".join(chunk['texts'])
        final_keywords = _filter_and_clean_keywords(chunk['keywords'], 5, tokenizer)
        section = chunk['section']
        context = Context(pdf_file_path.stem, 1, section, "Text", summary="")
        context.relevant_keywords = final_keywords
        augmented_content = f"{context.to_string()}\n\n[CONTENT]\n{final_text}"

        chunk_metadata = {
            "source": pdf_file_path.name, "company": company_ticker, "element_type": "Text",
            "section": section, "keywords": ", ".join(final_keywords)
        }
        chunk_metadata.update(extracted_metadata)
        final_documents.append(Document(page_content=augmented_content, metadata=chunk_metadata))

    return final_documents


# ==============================================================================
#  Stage 4: Final Summary Generation
# ==============================================================================

def _generate_final_summary(
    cleaned_md_content: str, table_contexts: List[Context], pdf_file_path: pathlib.Path, llm: ChatOpenAI
):
    """
    Generates and saves a high-level summary for the entire document.

    Args:
        cleaned_md_content: The full cleaned markdown text.
        table_contexts: A list of Context objects from processed tables.
        pdf_file_path: The path to the source PDF for naming the output file.
        llm: The language model to use for summarization.
    """
    if not table_contexts or not cleaned_md_content:
        return

    print("\n--- Generating High-Level Document Summary ---")
    try:
        document_summary = _get_document_summary(cleaned_md_content, table_contexts, llm)
        summary_output_path = SUMMARY_DIRECTORY / f"{pdf_file_path.stem}_summary.txt"
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(document_summary, encoding='utf-8')
        print(f"Document summary saved to: {summary_output_path}")
    except Exception as e:
        print(f"Failed to generate or save document summary: {e}")


def _get_document_summary(cleaned_md: str, table_contexts: List[Context], llm: ChatOpenAI) -> str:
    """
    Constructs a prompt and invokes an LLM to create a structured document summary.

    Args:
        cleaned_md: The full cleaned markdown content.
        table_contexts: A list of contexts, including table summaries.
        llm: The language model for generating the summary.

    Returns:
        The generated summary as a string.
    """
    document_outline_parts = [line for line in cleaned_md.splitlines() if line.startswith("##")]
    document_outline_parts.append("\nALL TABLE SUMMARIES ARE BELOW:")
    for context in table_contexts:
        if context.element_type.lower() == "table":
            document_outline_parts.append(context.summary)

    document_outline = "\n".join(document_outline_parts)
    template = DOCUMENT_SUMMARY_PROMPT

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"outline": document_outline})


# ==============================================================================
#  Utility and Helper Functions
# ==============================================================================

def _extract_json_from_string(text: str) -> str:
    """Extracts the first JSON object from a string, ignoring other text."""
    match = re.search(r'\{.*}', text, re.DOTALL)
    return match.group(0) if match else ""


def strip_thinking_tags(text: str) -> str:
    """Removes <think>...</think> tags from an LLM response."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def count_tokens(text: str, encoder: Encoding) -> int:
    """Counts tokens in a string using the provided tiktoken encoder."""
    return len(encoder.encode(text, disallowed_special=()))


def get_one_line_summary(table_text: str, section_title: str, parser: JsonOutputParser, llm: ChatOpenAI) -> Dict:
    """
    Generates a one-sentence summary of a financial table using an LLM.

    Args:
        table_text: The full markdown text of the table.
        section_title: The title of the section containing the table.
        parser: The LangChain JSON output parser.
        llm: The language model to use for summarization.

    Returns:
        A dictionary containing the summary.

    Raises:
        TableSummarizationError: If the LLM call or parsing fails.
    """
    master_prompt_template = TABLE_SUMMARY_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [("system", master_prompt_template), ("human", "Section: {section_title}\n\nTable:\n{table}")])
    chain = prompt | llm | (lambda x: strip_thinking_tags(x.content)) | parser

    try:
        return chain.invoke({"table": table_text, "section_title": section_title})
    except (OutputParserException, Exception) as e:
        raise TableSummarizationError(f"LLM call or parsing failed for table in section '{section_title}'") from e


def _filter_and_clean_keywords(candidates: Set[str], max_keywords: int, encoder: Encoding) -> List[str]:
    """Internal helper to filter, clean, and deduplicate keyword candidates."""
    final_keywords: List[str] = []
    seen_lower: Set[str] = set()

    # Sort by length to prioritize more specific, longer keywords
    sorted_candidates = sorted([str(c) for c in candidates], key=len, reverse=True)

    for keyword in sorted_candidates:
        kw_clean = keyword.strip(" “”)’'.,:()|").replace('’', "'").strip()
        kw_lower = kw_clean.lower()

        # Filter out stopwords, short words, numbers, and long tokens
        if not kw_clean or len(kw_clean) < 4 or kw_lower in STOP_WORDS or any(char.isdigit() for char in kw_clean) or count_tokens(kw_clean, encoder) > 7:
            continue

        # Filter out keywords that are substrings of already added keywords
        if not any(kw_lower in seen for seen in seen_lower):
            final_keywords.append(re.sub(r'^(?:A|An|The|Their)\s+', '', kw_clean, flags=re.IGNORECASE))
            seen_lower.add(kw_lower)

        if len(final_keywords) >= max_keywords:
            break
    return final_keywords


def batch_extract_nlp_keywords(
    texts: List[str], nlp: Language, encoder: Encoding, max_keywords_per_item: int = 5
) -> List[List[str]]:
    """
    Extracts keywords from a batch of texts using preloaded spaCy and tiktoken models.

    Args:
        texts: A list of text strings to process.
        nlp: The preloaded spaCy language model.
        encoder: The preloaded tiktoken encoder.
        max_keywords_per_item: The maximum number of keywords to extract per text.

    Returns:
        A list of lists, where each inner list contains the keywords for the corresponding text.
    """
    all_results = []
    # Basic cleaning before processing
    cleaned_texts = [re.sub(r'^\s*([a-zA-Z0-9]+\.|-|\*)\s+', '', item, flags=re.MULTILINE) for item in texts]
    cleaned_texts = [' '.join(text.split()) for text in cleaned_texts]

    # Use nlp.pipe for efficient batch processing
    docs = nlp.pipe(cleaned_texts, batch_size=500)

    for doc in docs:
        candidates = {chunk.text for chunk in doc.noun_chunks} | {ent.text for ent in doc.ents}
        keywords = _filter_and_clean_keywords(candidates, max_keywords_per_item, encoder)
        all_results.append(keywords)

    return all_results