import pathlib
import re
from typing import List, Callable, Dict, Tuple

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table


def _heuristic_reject_boilerplate(table_element: Table, table_text: str) -> bool:
    negative_keywords = ["exhibit", "item", "risk factors", "forward-looking statements"]
    return not any(keyword in table_text.lower() for keyword in negative_keywords)


def _get_structural_score(table_element: Table, table_text: str) -> Tuple[int, str]:
    score = 0
    reasons = []

    if '<th>' in table_text.lower():
        score += 25
        reasons.append("Has Headers (+25)")

    num_rows = table_text.count('<tr')
    num_cells = table_text.count('<td')
    if num_rows > 3 and num_cells > (num_rows * 1.5):
        score += 20
        reasons.append(f"Good Dims ({num_rows}r, {num_cells}c) (+20)")

    num_digits = sum(c.isdigit() for c in table_text)
    total_chars = len(table_text) if len(table_text) > 0 else 1
    digit_ratio = num_digits / total_chars
    if digit_ratio > 0.10:
        density_score = int(digit_ratio * 50)
        score += density_score
        reasons.append(f"Num Density ({digit_ratio:.1%}) (+{density_score})")

    return score, ", ".join(reasons)


def _get_cell_content_ratio(table_element: Table, table_text: str) -> float:
    cells = re.findall(r'<td.*?>(.*?)</td>', table_text, re.DOTALL)
    if not cells: return 0.0

    data_cell_count = 0
    for cell in cells:
        clean_cell = re.sub(r'<.*?>', '', cell).strip()
        if not clean_cell: continue
        if re.fullmatch(r'[\$\(]?\s*[\d,]+(\.\d+)?\s*[\)%]?', clean_cell):
            data_cell_count += 1

    return data_cell_count / len(cells)


def heuristic_ensemble_model(table_element: Table, table_text: str) -> Tuple[bool, str]:
    # Step 1: Immediately reject boilerplate
    if not _heuristic_reject_boilerplate(table_element, table_text):
        return False, "Rejected as boilerplate"

    # Step 2: Calculate the base structural score
    structural_score, structural_reasons = _get_structural_score(table_element, table_text)

    # Step 3: Calculate the data cell ratio to use as a multiplier
    data_cell_ratio = _get_cell_content_ratio(table_element, table_text)

    # Step 4: Combine scores. The data ratio acts as a powerful multiplier.
    # A table with no data cells gets its structural score halved.
    # A table with 100% data cells gets its score boosted by 50%.
    final_score = structural_score * (0.5 + data_cell_ratio)

    SCORE_THRESHOLD = 35

    reason = (
        f"Final Score: {final_score:.0f} "
        f"(Structural: {structural_score}, Cell Ratio Multiplier: {0.5 + data_cell_ratio:.2f})"
    )

    return final_score >= SCORE_THRESHOLD, reason


# ==============================================================================
# MAIN TESTBED CLASS (for diagnostics)
# ==============================================================================

class HeuristicTestbed:
    def __init__(self, file_path: str):
        self.file_path = pathlib.Path(file_path)
        self.tables_data = []

    def _partition_and_extract_tables(self):
        print(f"--- Partitioning Document: {self.file_path.name} ---")
        if self.tables_data: return

        try:
            elements = partition_pdf(self.file_path, strategy="hi_res", infer_table_structure=True)
            table_elements = [el for el in elements if isinstance(el, Table)]

            for i, table_el in enumerate(table_elements):
                self.tables_data.append({
                    "index": i + 1,
                    "page": table_el.metadata.page_number,
                    "element": table_el,
                    "text": table_el.metadata.text_as_html if hasattr(table_el.metadata,
                                                                      'text_as_html') else table_el.text
                })
            print(f"Successfully extracted {len(self.tables_data)} tables.")
        except Exception as e:
            print(f"\n--- 🔴 An error occurred during PDF partitioning ---")
            print(e)

    def run_and_report(self, heuristics: Dict[str, Callable]):
        self._partition_and_extract_tables()
        if not self.tables_data: return

        print(f"\n--- Running {len(heuristics)} Heuristics ---")

        results: Dict[str, List[Dict]] = {name: [] for name in heuristics.keys()}

        for table_data in self.tables_data:
            for name, func in heuristics.items():
                is_hit, reason = func(table_data["element"], table_data["text"])
                if is_hit:
                    results[name].append({
                        "index": table_data["index"],
                        "page": table_data["page"],
                        "reason": reason,
                        "content": table_data["text"]
                    })

        self._print_report(results)

    def _print_report(self, results: Dict[str, List[Dict]]):
        print("\n" + "=" * 80)
        print("--- HEURISTIC TESTBED SUMMARY ---")
        print("=" * 80)

        for name, hits in results.items():
            print(f"\n\n--- Heuristic: '{name}' ---")
            print(f"   Total Hits: {len(hits)}")
            print("-" * 50)

            if not hits:
                print("   No tables matched this heuristic.")
                continue

            for hit in hits:
                print(f"✅ HIT (Table #{hit['index']} on Page {hit['page']})")
                print(f"   Reason: {hit['reason']}")
                cleaned_content = re.sub(r'<.*?>', ' ', hit['content']).replace('\n', ' ').strip()
                cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
                print(f"   Content: {cleaned_content[:250]}...")


if __name__ == "__main__":

    heuristics_to_test = {
        "NEW Ensemble Model": heuristic_ensemble_model,
    }

    target_pdf_path = "../reports/aapl-20230930.pdf"

    if not pathlib.Path(target_pdf_path).exists():
        print(f"Error: Test file not found at '{target_pdf_path}'")
    else:
        testbed = HeuristicTestbed(file_path=target_pdf_path)
        testbed.run_and_report(heuristics=heuristics_to_test)
