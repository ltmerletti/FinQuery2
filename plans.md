TODO:
- Implement a deduplication system somehow
- Attempt MRI (if time permits)
- Finish pipeline to LLM (Making sure to remove all context *before* sending it to LLM to preserve tokens)
- Make 100 total questions with answers and have an AI like gemini evaluate them for a test. 
- Make this PDF breakdown thing (chunker, etc.) into a library
- Try to remove <30 character chunks to see why they exist

---

I could use semantic chunking for non-tables and separate tables from that since there are already the section headers in [CONTEXT]
This could use a langchain semantic chunker? https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html

---

Plan for better chunking (help from Gemini):
- It iterates through your list of Document chunks in order.
- When it finds an element it identifies as a Table, it "pauses."
- It then looks at the chunks immediately before and after the table to see if they are a Title or a Footnote.
- If they are, it combines their text content into a new, single "mega-chunk."
- It then skips ahead in the list past all the chunks it just combined and continues the process.

--- 

Plan for table cleaning (consider it):

### Guiding Philosophy: A Multi-Stage Cleaning Pipeline

A robust system isn't a single, magical function. It's a pipeline of sequential, independent cleaning steps. Each step handles one specific class of error, and if it fails, it doesn't break the entire process. This makes the system resilient, easier to debug, and less brittle.

Our pipeline will be:
1.  **Pre-Processing:** Normalize the raw text string.
2.  **Structural Repair:** Rebuild the table's grid structure, even if the Markdown is broken.
3.  **Content Cleaning:** Clean the text *inside* each individual cell.
4.  **Reconstruction & Validation:** Re-assemble the cleaned data into perfect Markdown and validate its integrity.

---

### The Four-Stage Table Cleaning Plan

Here is the precise plan, with code examples for each stage. You would build this into a `TableCleaner` class.

#### Stage 1: Pre-Processing (Normalize the Raw String)
This stage fixes simple, global text issues before we attempt to parse the structure. It's the cheapest and safest place to make corrections.

**What to do:**
*   Replace common OCR errors (e.g., `et sales` -> `Net sales`).
*   Standardize whitespace (replace multiple spaces with a single space, but preserve newlines).
*   Fix jumbled characters that often appear together (like a number followed by a currency symbol).

```python
def _normalize_text(self, text: str) -> str:
    """Fixes common, document-wide text errors."""
    corrections = {
        "et sales": "Net sales",
        "ost of sales": "Cost of sales",
        "perating expenses": "Operating expenses",
        "perating income": "Operating income",
        # Add more common OCR errors here
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    
    # Add space between numbers and currency symbols (e.g., "123$ -> 123 $")
    import re
    text = re.sub(r'(\d)([€$¢])', r'\1 \2', text)
    
    return text
```

#### Stage 2: Structural Repair (Rebuild the Grid)
This is the most critical step. **Do not trust the incoming Markdown pipes (`|`)**. They are often missing or misaligned. Instead, deduce the structure from more reliable signals like newlines and consistent spacing.

**What to do:**
1.  Split the text block into rows using newline characters.
2.  For each row, try to split it into columns using the pipe `|`.
3.  **If that fails (i.e., you only get one column), use a robust fallback:** Split the row by sequences of two or more spaces. This is a powerful heuristic for poorly formatted tables.
4.  The result should be a list of lists, representing the grid (`List[List[str]]`).

```python
def _repair_structure(self, text: str) -> List[List[str]]:
    """Rebuilds the table grid, ignoring faulty Markdown."""
    import re
    grid = []
    rows = text.strip().split('\n')
    
    for row_text in rows:
        row_text = row_text.strip()
        if not row_text:
            continue
        
        # First, try splitting by the markdown pipe
        columns = [cell.strip() for cell in row_text.split('|')]
        
        # If splitting by pipe results in only one or two columns (likely a failed parse)
        # and it's not the markdown header line, use the whitespace fallback.
        if len(columns) <= 2 and not re.match(r'^[:\s|-]+$', row_text):
            # Fallback: split by 2 or more spaces
            columns = [cell.strip() for cell in re.split(r'\s{2,}', row_text)]
            
        # Filter out empty columns that can result from leading/trailing pipes
        columns = [cell for cell in columns if cell]
        
        if columns:
            grid.append(columns)
            
    return grid
```

#### Stage 3: Content Cleaning (Clean Each Cell)
Now that you have a reliable grid structure, iterate through every cell and clean its specific content.

**What to do:**
*   Remove leading/trailing whitespace from each cell.
*   Standardize financial negatives (e.g., `(1,234)` -> `-1234`).
*   Remove artifacts like random asterisks or stray characters.

```python
def _clean_cell(self, cell_text: str) -> str:
    """Cleans the content within a single table cell."""
    text = cell_text.strip()
    
    # Standardize financial negatives: (123) -> -123
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
        
    # Remove commas from numbers to make them easier to parse as floats
    text = text.replace(',', '')
    
    # Remove any other artifacts you commonly see
    text = text.replace('§$', '$').replace('“', '')
    
    return text

# In the main function, you'd apply this to the grid:
# cleaned_grid = [[self._clean_cell(cell) for cell in row] for row in grid]
```

#### Stage 4: Reconstruction and Validation
Finally, rebuild the cleaned grid into perfect, valid Markdown and perform a final check.

**What to do:**
1.  Determine the number of columns from the longest row in your cleaned grid.
2.  Rebuild each row as a pipe-delimited string, ensuring every row has the same number of columns by padding with empty cells if necessary.
3.  **Validation:** If the number of columns is inconsistent across rows *after* cleaning, the table is likely too complex or malformed to be fixed automatically. In this case, it is safer to **return the original, unprocessed table string with a warning log** than to return a confidently wrong, broken table. This is a key part of making the system "not brittle."

```python
def _reconstruct_and_validate(self, grid: List[List[str]]) -> str:
    """Rebuilds the table into valid Markdown and validates its structure."""
    if not grid:
        return ""

    # Find the maximum number of columns required for the table
    num_columns = max(len(row) for row in grid) if grid else 0

    # Basic validation: If there are no columns or rows, it's not a table.
    if num_columns == 0:
        return "" # Or return the original string

    # Rebuild the table with consistent column counts
    markdown_rows = []
    for row in grid:
        # Pad rows that are shorter than the max column count
        padded_row = row + [''] * (num_columns - len(row))
        markdown_rows.append('| ' + ' | '.join(padded_row) + ' |')

    # Add the Markdown header separator if it's missing
    if len(markdown_rows) > 1 and not re.match(r'^[:\s|-]+$', markdown_rows[1]):
        separator = '|' + ' :--- |' * num_columns
        markdown_rows.insert(1, separator)
        
    # Final Validation: If a row has too many columns, something is wrong.
    # This is a good place to log a warning.
    for row in grid:
        if len(row) > num_columns:
            # LOG.warning("Table row has inconsistent column count. Returning original.")
            # In a real system, you'd return the original text here as a fallback.
            pass

    return '\n'.join(markdown_rows)
```

By combining these four stages into a single `TableCleaner` class, you create a robust, testable, and extensible system for fixing the table data your RAG pipeline depends on.