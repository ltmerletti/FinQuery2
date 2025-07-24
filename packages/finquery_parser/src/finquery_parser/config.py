import re

# Experiment with these first two and see what works best for your system. For the financial documents,
# The content tends to be semantically dense, so I recommend smaller chunks.
MAX_CHUNK_TOKENS = 256
MIN_CHUNK_TOKENS = 175

# The cover page token count doesn't matter too much. Look at your documents and find a rough estimate of the
# tokens used for the token count
COVER_PAGE_TOKEN_COUNT = 400

# This is the first bit of the document that will be sent for metadata extraction. 2000 seems to be the sweet spot.
TIER_1_SNIPPET_TOKEN_LIMIT = 2000

# 600 chars is the max preface length for a table. Beyond that, it's likely a paragraph.
MAX_PREFACE_LENGTH = 600

VISION_MODEL_ID = "unsloth/granite-vision-3.2-2b-unsloth-bnb-4bit"

MARKDOWN_CLEANING_PATTERNS = [
    # Get rid of table of contents mentions, sometimes a link back
    re.compile(r"^## table of contents\s*$", re.IGNORECASE),
    # Get rid of page number artifacts
    re.compile(r"^\s*(\|?\s*page\s*\|?\s*)?\d+(\s*of\s*\d+)?\s*\|?\s*$", re.IGNORECASE),
    # Get rid of page number artifacts
    re.compile(r"^([*\-]) \[PAGE \d+]\(#\d+\)\s*$"),
    # For 10-Ks specifically, remove the artifact at the bottom
    re.compile(r".*Form 10-K\s*\|\s*\d+\s*$", re.IGNORECASE | re.MULTILINE)
]

# Words not to be taken as keywords, since spaCy could perform badly at times
STOP_WORDS = {
    'a', 'an', 'and', 'the', 'is', 'it', 'in', 'on', 'for', 'of', 'as', 'to', 'inc', 'was', 'were', 'by', 'with', 'or',
    'at', 'from', 'that', 'this', 'llc', 'ltd', 'company', 'corp', 'about', 'after', 'all', 'also', 'been', 'because',
    'but', 'can', 'could', 'did', 'do', 'due', 'has', 'had', 'have', 'how', 'however', 'into', 'its', 'just', 'may',
    'most', 'must', 'not', 'other', 'our', 'out', 'over', 'said', 'should', 'so', 'some', 'such', 'than', 'then',
    'there', 'these', 'they', 'through', 'under', 'upon', 'use', 'used', 'using', 'various', 'very', 'we', 'what',
    'when', 'where', 'which', 'while', 'who', 'why', 'will', 'would', 'you', 'your', 'notes', 'note', 'see', 'title',
    'part', 'item', 'items', 'page', 'the company', 'apple inc', 'registrant', 'thereof', 'thereto', 'therein',
    'thereon', 'hereto', 'hereof', 'herein', 'hereinafter', 'pursuant', 'including', 'certain', 'related',
    'primarily', 'approximately', 'significant', 'generally', 'thereunto', 'therewith'
}

# Words to use when detecting a preface; check your documents and change as you see fit
PREFACE_KEYWORDS = [
    'the following table', 'consisted of the following', 'as follows:', 'were as follows', 'summarizes', 'presents',
    'sets forth', '(in thousands', '(in millions', 'except per share'
]

TABLE_SUMMARY_PROMPT = """You are an Expert Financial Analyst creating single-sentence summaries of financial tables for a RAG system. Your response must be a single, clean JSON object with one key: "summary".
**RULES FOR THE SUMMARY SENTENCE:**
1.  **Analyst's Perspective:** Describe what the table *allows an analyst to do* (e.g., "analyze trends," "assess financial health").
2.  **Keyword-in-Sentence:** Weave the table's most important headers into the sentence.
3.  **Combine Specifics with Concepts:** Merge specific entities (e.g., `iPhone`, `2023`) with financial concepts (e.g., `revenue streams`).
4.  **No Numerical Data:** Do not include specific numbers or dollar amounts.
5.  **Concise but Dense:** The sentence should be a single, flowing thought packed with searchable context.
---
\\no_think"""

DOCUMENT_SUMMARY_PROMPT = """You are an expert document analyst. Your task is to create a dense, structured summary of a document that acts as a "meta-document" for retrieval.
Based on the document's headings and table summaries provided, generate a summary.
**Instructions:**
1.  Identify all major topics, entities, and concepts from the outline.
2.  Create a "Keyword Index" listing key terms under these generic Markdown headings: `### Key Topics & Concepts:`, `### Key Entities:`, `### Geographic Locations:`, `### Discussed Risks:`.
3.  Create a brief "Analytical Summary" (1-2 sentences) for the most important categories.
---
**DOCUMENT OUTLINE AND TABLE SUMMARIES:**
{outline}
---
**STRUCTURED DOCUMENT SUMMARY:**"""

METADATA_EXTRACTION_PROMPT = """You are a meticulous data extraction specialist AI. Your task is to extract specific pieces of information from a document snippet and populate a given JSON schema.
**Instructions:**
1.  Carefully review the DOCUMENT SNIPPET.
2.  Fill in the values for every field listed in the METADATA SCHEMA.
3.  **Crucially: If you cannot confidently find a value for a specific field, you MUST use `null` as its value. Do not guess or infer information.**
4.  Your entire response MUST be a single, valid JSON object that exactly matches the METADATA SCHEMA.
---
**METADATA SCHEMA:**
{schema}
---
**DOCUMENT SNIPPET:**
{snippet}
---
**JSON RESPONSE:**
"""

DOCUMENT_TYPE_AND_SCHEMA_PROMPT = """You are a data architect AI. Your task is to analyze a new document's structure and cover page to decide if it matches a known document type or if a new type needs to be created.

You will be given a list of KNOWN_DOC_TYPES, the DOCUMENT_OUTLINE (headings), and the COVER_PAGE_CONTENT (the text from the first page). The cover page is often the most reliable source for identifying information.

**Your Instructions (Follow Rigorously):**
1.  **Analyze Context:** Analyze the COVER_PAGE_CONTENT and DOCUMENT_OUTLINE to hypothesize the document's type.
2.  **Compare Against Known Types:** Compare your hypothesis against the `type_name` and `metadata_schema` of every entry in KNOWN_DOC_TYPES.
3.  **Decision Logic:**
    * **IF** you find a strong match, your action is "match".
    * **ELSE (IF NO MATCH IS FOUND),** your action is "create". This is the default. If the document type is not in KNOWN_DOC_TYPES, you MUST create it.
4.  **Schema Generation (for "create"):** Identify 3-5 critical metadata fields. Prioritize unique identifiers (e.g., company name, date, report number). Use `snake_case` for field names.
5.  **Final Output:** Your entire response MUST be a single, valid JSON object.
    * If `action` is "match", the JSON **must** include the `action`, `type_name`, and `type_id` of the matched type.
    * If `action` is "create", the JSON **must** include the `action`, `new_type_name`, and `new_metadata_schema`.

---
**EXAMPLE 1: MATCH**
*INPUT:*
- KNOWN_DOC_TYPES: `[{{"id": 1, "type_name": "SEC 10-K Annual Report", "metadata_schema": {{"company_name": "text", "fiscal_year": "integer"}}}}]`
- COVER_PAGE_CONTENT: "FORM 10-K... for the fiscal year ended September 30, 2023... APPLE INC."
*CORRECT JSON RESPONSE:*
```json
{{
  "action": "match",
  "type_name": "SEC 10-K Annual Report",
  "type_id": 1
}}
```
---
**EXAMPLE 2: CREATE**
*INPUT:*
- KNOWN_DOC_TYPES: `[]`
- COVER_PAGE_CONTENT: "FORM 10-K... for the fiscal year ended September 30, 2023... APPLE INC."
*CORRECT JSON RESPONSE:*
```json
{{
  "action": "create",
  "new_type_name": "SEC 10-K Annual Report",
  "new_metadata_schema": {{
    "company_name": "text",
    "fiscal_year": "integer",
    "cik_number": "text"
  }}
}}
```
---
**TASK:**
---
**KNOWN_DOC_TYPES:**
{known_doc_types_json}
---
**COVER_PAGE_CONTENT:**
{cover_page_text}
---
**DOCUMENT_OUTLINE:**
{document_outline}
---
**JSON RESPONSE:**"""