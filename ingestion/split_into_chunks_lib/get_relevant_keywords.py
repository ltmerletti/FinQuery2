import re
from typing import List

import bs4
from unstructured.documents.elements import Table, Element

from ingestion.split_into_chunks_lib.Context import Context

# A comprehensive set of stop words, including common financial boilerplate
STOP_WORDS = {
    'a', 'an', 'and', 'the', 'is', 'it', 'in', 'on', 'for', 'of', 'as', 'to', 'inc',
    'was', 'were', 'by', 'with', 'or', 'at', 'from', 'that', 'this', 'llc', 'ltd',
    'company', 'corp', 'about', 'after', 'all', 'also', 'been', 'because', 'but',
    'can', 'could', 'did', 'do', 'due', 'has', 'had', 'have', 'how', 'however',
    'into', 'its', 'just', 'may', 'most', 'must', 'not', 'other', 'our', 'out',
    'over', 'said', 'should', 'so', 'some', 'such', 'than', 'then', 'there',
    'these', 'they', 'through', 'under', 'upon', 'use', 'used', 'using', 'various',
    'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
    'why', 'will', 'would', 'you', 'your', 'notes', 'note', 'see', 'title', 'part',
    'item', 'items', 'page', 'inc', 'corp', 'ltd', 'llc', 'the company', 'apple inc',
    'registrant', 'thereof', 'thereto', 'therein', 'thereon', 'hereto', 'hereof',
    'herein', 'hereinafter', 'pursuant', 'including', 'certain', 'related',
    'primarily', 'approximately', 'significant', 'generally'
}


def get_relevant_keywords(element: Element, context: Context, max_keywords: int = 15) -> List[str]:
    """
    Extracts high-quality keywords using refined, zero-dependency heuristics.
    """
    if not element or not element.text.strip():
        return []

    text_to_process = element.text
    candidates = set()

    # Heuristic 1: Specialized Logic for Tables
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

    # Heuristic 2: Context-Aware Logic for All Elements
    if context.section_title:
        candidates.add(context.section_title)

    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text_to_process)
    candidates.update(capitalized_phrases)

    acronyms = re.findall(r'\b[A-Z]{2,5}\b', text_to_process)
    candidates.update(acronyms)

    # --- Final Cleaning and Ranking ---
    final_keywords = []
    seen_lower = set()

    sorted_candidates = sorted(list(candidates), key=len, reverse=True)

    for keyword in sorted_candidates:

        if not isinstance(keyword, str):
            continue

        kw_clean = keyword.strip(" “”)’'.,:()").replace('’', "'")
        kw_lower = kw_clean.lower()

        if not kw_clean or len(kw_clean) <= 3 or kw_lower in STOP_WORDS or kw_lower.isdigit():
            continue

        is_redundant = False
        for seen in seen_lower:
            if kw_lower in seen:
                is_redundant = True
                break
        if is_redundant:
            continue

        if kw_lower not in seen_lower:
            final_keywords.append(kw_clean)
            seen_lower.add(kw_lower)

        if len(final_keywords) >= max_keywords:
            break

    return final_keywords
