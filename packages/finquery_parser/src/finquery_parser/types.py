from docling_core.types.doc import (TableItem as DoclingTableItem, TextItem as DoclingTextItem,
                                    DoclingDocument as DoclingDoc, )


class Context:
    def __init__(self, pdf_title, page_number, section_title, element_type, summary, table_prefix=""):
        self.pdf_title = pdf_title
        self.page_number = page_number
        self.section_title = section_title
        self.element_type = element_type
        self.summary = summary
        self.relevant_keywords = []
        self.table_prefix = table_prefix

    def to_string(self):
        return f"[CONTEXT]\nPDF Title: {self.pdf_title}\nSection: {self.section_title}\nKeywords: {', '.join(self.relevant_keywords)}\nSummary: {self.summary}"


class TableSummary:
    pass


class TableSummarizationError(Exception):
    """Custom exception for errors during the LLM table summarization step."""
    pass


class _DoclingElementAdapter:
    """Base adapter to make Docling items compatible with dependent functions."""

    def __init__(self, item: DoclingTextItem | DoclingTableItem, doc: DoclingDoc):
        self._item = item
        self._doc = doc
        self.id = id(item)

    @property
    def metadata(self):
        """Creates a mock metadata object with page_number."""

        class MockMetadata:
            pass

        meta = MockMetadata()
        if self._item.prov and self._item.prov[0]:
            meta.page_number = self._item.prov[0].page_no
        else:
            meta.page_number = None
        return meta


class DoclingTableAdapter(_DoclingElementAdapter):
    """Adapter for Docling's TableItem."""

    @property
    def text(self) -> str:
        """Returns a markdown string representation of the table."""
        try:
            return self._item.export_to_markdown(doc=self._doc)
        except Exception:
            return ""


class DoclingTextAdapter(_DoclingElementAdapter):
    """Adapter for Docling's TextItem."""

    @property
    def text(self) -> str:
        """Returns the text content of the item."""
        return self._item.text
