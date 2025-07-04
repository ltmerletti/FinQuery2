from typing import Optional


class Context:
    def __init__(
            self,
            pdf_title: str = "",
            page_number: int = 0,
            section_title: str = "",
            relevant_keywords: list[str] = None,
            element_type: str = "",
            summary: Optional[str] = None
    ):
        self.pdf_title = pdf_title
        self.page_number = page_number
        self.section_title = section_title
        self.relevant_keywords = relevant_keywords or []
        self.element_type = element_type
        self.summary = summary or None

    def to_string(self):
        return f"""[CONTEXT]
PDF Title: {self.pdf_title}
Page Number: {self.page_number}
Section Title: {self.section_title}
Element Type: {self.element_type}
Relevant Keywords: {str(self.relevant_keywords)}
{'Summary: ' + self.summary if self.summary else ''}"""

    def to_dict(self):
        return {
            "pdf_title": self.pdf_title,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "relevant_keywords": self.relevant_keywords,
            "summary": self.summary,
            "element_type": self.element_type
        }
