import pathlib
import re
from typing import List, Tuple

from unstructured.documents.elements import Table, Text, Title
from unstructured.partition.pdf import partition_pdf

from ingestion.split_into_chunks_lib.Context import Context
from ingestion.split_into_chunks_lib.get_one_line_summary import get_one_line_summary
from ingestion.split_into_chunks_lib.get_relevant_keywords import get_relevant_keywords
from ingestion.split_into_chunks_lib.is_table_empty import is_table_functionally_empty

# this pattern gets rid of the 'Apple Inc. | 2023 Form 10-K | 21' stuff we don't want
JUNK_FOOTER_PATTERN = re.compile(r'^.*Form 10-K\s*\|\s*\d+\s*$', re.IGNORECASE | re.MULTILINE)
# this pattern helps identify and discard titles that are just form-like checkboxes.
CHECKBOX_PATTERN = re.compile(r'Yes\s*☒\s*No\s*☐', re.IGNORECASE)
# this filters out sec links
SEC_LINK_PATTERN = re.compile(r'https?://www\.sec\.gov/Archives/edgar/data/.*\.htm', re.IGNORECASE)


def partition_and_separate_elements(pdf_file_path: pathlib.Path) -> Tuple[
    Tuple[List[Table], List[Context]], Tuple[List[Text], List[Context]]]:
    print(f"Partitioning document: {pdf_file_path}")

    if not str(pdf_file_path).endswith(".pdf"):
        return ([], []), ([], [])

    elements = list(partition_pdf(pdf_file_path, strategy="hi_res", infer_table_structure=True))
    print(f"\n--- Found {len(elements)} raw elements. Processing sequentially... ---")

    table_elements, table_contextss = [], []
    text_elements, text_contextss = [], []

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

        contextt: Context = Context(
            pdf_title=pdf_file_path.stem,
            page_number=getattr(el.metadata, 'page_number', None),
            section_title=current_section_title,
            element_type="Table" if isinstance(el, Table) else "Text",
            summary=""
        )

        contextt.relevant_keywords = get_relevant_keywords(el, contextt, 5)

        if contextt.element_type == "Table":
            one_sentence_summary = get_one_line_summary(el.text, contextt.section_title)
            contextt.summary = one_sentence_summary['summary']

        if isinstance(el, Table):
            if not is_table_functionally_empty(getattr(el.metadata, 'text_as_html', None)):
                table_elements.append(el)
                table_contextss.append(contextt)
        else:
            if len(el.text.strip()) > 25:
                text_elements.append(el)
                text_contextss.append(contextt)

    print(
        f"--- Finished processing. Found {len(table_elements)} tables and {len(text_elements)} text elements. ---")

    return (table_elements, table_contextss), (text_elements, text_contextss)
