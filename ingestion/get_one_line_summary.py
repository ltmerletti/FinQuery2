import os
import re
from typing import Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class TableSummary(BaseModel):
    summary: str = Field(description="A single-sentence, non-quantitative description of the table's contents.")


parser = JsonOutputParser(pydantic_object=TableSummary)


def get_one_line_summary(table_text: str, section_title: str) -> Dict:
    load_dotenv()
    llm = ChatOpenAI(
        model=os.getenv("LMSTUDIO_MODEL_NAME"),
        base_url=os.getenv("LMSTUDIO_BASE_URL"),
        api_key=os.getenv("LMSTUDIO_API_KEY")
    )

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
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", master_prompt_template),
        ("human", "Section: {section_title}\n\nTable:\n{table}")
    ])

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"table": table_text, "section_title": section_title})
    except Exception as e:
        print(f"Error: Could not process LLM output. Error: {e}")
        result = {"summary": "Error summarizing table."}

    return result


if __name__ == "__main__":
    example_table = """
| | 2023 | 2022 |
| :--- | :--- | :--- |
| Total current assets | 143,566 | 135,405 |
| Total non-current assets | 209,017 | 217,350 |
| **Total assets** | **352,583** | **352,755** |
| Total current liabilities | 145,308 | 153,982 |
| Total non-current liabilities | 145,129 | 148,101 |
| **Total liabilities** | **290,437** | **302,083** |
| **Total shareholders’ equity** | **62,146** | **50,672** |
"""
    example_section_title = "CONSOLIDATED BALANCE SHEETS"

    summary_json = get_one_line_summary(example_table, example_section_title)

    print("\n--- LLM Output ---")
    print(summary_json)