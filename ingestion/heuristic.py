import re
from typing import Tuple

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

def isHighValue(table_element, table_text):
    is_hit, _ = heuristic_ensemble_model(table_element, table_text)
    return is_hit
