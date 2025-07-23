import json
from typing import Set, List, Dict, Optional, TypedDict, Protocol

import psycopg2
from docling_core.types.doc import (TableItem as DoclingTableItem, TextItem as DoclingTextItem,
                                    DoclingDocument as DoclingDoc)


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

class PDFConversionError(Exception):
    """Custom exception for failures during the PDF to Markdown conversion process."""
    pass


class MetadataExtractionError(Exception):
    """Custom exception for failures during the metadata extraction process."""
    pass


class Header(TypedDict):
    """Represents a markdown header with its level and title."""
    level: int
    title: str


class ProseElement(TypedDict):
    """Represents a block of text content from the document."""
    order: int
    content: str
    headers: List[Header]
    keywords: List[str]


class TableElement(TypedDict):
    """Represents a table extracted from the document."""
    order: int
    content: str
    headers: List[Header]
    preface: Optional[str]


class IntermediateChunk(TypedDict):
    """Represents a temporary grouping of text before final processing."""
    texts: List[str]
    keywords: Set[str]
    section: str


class DatabaseInterface(Protocol):
    """
    Defines the protocol for a database connection object.
    This allows the parsing library to be completely decoupled from the
    specific database implementation (e.g., PostgreSQL, SQLite).
    """
    def get_known_doc_types(self) -> List[Dict]:
        """Fetches all known document types and their schemas from the database."""
        ...

    def create_doc_type(self, type_name: str, schema: Dict) -> Dict:
        """Creates a new document type in the database and returns it."""
        ...

class PostgresDBConnector:
    """
    A connector for a PostgreSQL database that implements the DatabaseInterface.
    """
    def __init__(self, dbname, user, password, host, port):
        print("Initializing PostgreSQL DB Connector...")
        self.conn_string = f"dbname='{dbname}' user='{user}' password='{password}' host='{host}' port='{port}'"
        self._conn = None
        try:
            self._conn = psycopg2.connect(self.conn_string)
            print("DB connection successful.")
            self._setup_tables()
        except psycopg2.OperationalError as e:
            print(f"DATABASE CONNECTION FAILED: {e}")
            print("Please check your connection details and ensure the database is running.")
            raise

    def _setup_tables(self):
        """Creates the necessary tables if they don't already exist."""
        create_types_table_sql = """
        CREATE TABLE IF NOT EXISTS document_types (
            id SERIAL PRIMARY KEY,
            type_name VARCHAR(256) NOT NULL UNIQUE,
            metadata_schema JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self._conn.cursor() as cur:
            cur.execute(create_types_table_sql)
        self._conn.commit()
        print("Database tables verified.")

    def get_known_doc_types(self) -> List[Dict]:
        """Fetches all known document types and their schemas from the database."""
        print("DB -> Fetching known document types...")
        with self._conn.cursor() as cur:
            cur.execute("SELECT id, type_name, metadata_schema FROM document_types ORDER BY id;")
            rows = cur.fetchall()
            doc_types = [
                {"id": row[0], "type_name": row[1], "metadata_schema": row[2]}
                for row in rows
            ]
            return doc_types

    def create_doc_type(self, type_name: str, schema: Dict) -> Dict:
        """Creates a new document type in the database and returns it."""
        print(f"DB -> Creating new document type: '{type_name}'")
        sql = """
        INSERT INTO document_types (type_name, metadata_schema)
        VALUES (%s, %s)
        RETURNING id, type_name, metadata_schema;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (type_name, json.dumps(schema)))
            new_row = cur.fetchone()
            self._conn.commit()
            if new_row:
                return {"id": new_row[0], "type_name": new_row[1], "metadata_schema": new_row[2]}
        return {}

    def close(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            print("Database connection closed.")