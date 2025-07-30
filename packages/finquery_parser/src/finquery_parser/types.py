import json
from typing import Set, List, Dict, Optional, TypedDict, Protocol, Any

import psycopg2
from docling_core.types.doc import (TableItem as DoclingTableItem, TextItem as DoclingTextItem,
                                    DoclingDocument as DoclingDoc)
from langchain_core.documents import Document


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

class DocumentList(List[Document]):
    """
    A custom list subclass that can carry extra metadata.
    It behaves exactly like a list but has an additional `.metadata` attribute.
    """
    def __init__(self, *args, metadata: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata or {}

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
        """Initializes the connection and sets up database tables."""
        print("Initializing PostgreSQL DB Connector...")
        self.conn_string = f"dbname='{dbname}' user='{user}' password='{password}' host='{host}' port='{port}'"
        self._conn = None
        try:
            self._conn = psycopg2.connect(self.conn_string)
            print("DB connection successful.")
            self._setup_tables()
        except psycopg2.OperationalError as e:
            print(f"DATABASE CONNECTION FAILED: {e}")
            print("Please make sure the database is running.")
            raise

    def _setup_tables(self):
        """
        Creates the tables if they don't exist.
        - document_types: Stores schemas for different types of documents.
        - documents: Stores a record for each processed file.
        - document_metadata_values: Stores the extracted key-value pairs for each document.
        """
        sql_statements = [
            """
            CREATE TABLE IF NOT EXISTS document_types (
                id SERIAL PRIMARY KEY,
                type_name VARCHAR(256) NOT NULL UNIQUE,
                metadata_schema JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(512) NOT NULL UNIQUE,
                document_type_id INTEGER REFERENCES document_types(id) ON DELETE SET NULL,
                processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS document_metadata_values (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                meta_key VARCHAR(128) NOT NULL,
                meta_value TEXT NOT NULL,
                UNIQUE (document_id, meta_key)
            );
            """
        ]
        with self._conn.cursor() as cur:
            for statement in sql_statements:
                cur.execute(statement)
        self._conn.commit()
        print("Database tables verified and set up.")

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
        ON CONFLICT (type_name) DO NOTHING
        RETURNING id, type_name, metadata_schema;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (type_name, json.dumps(schema)))
            new_row = cur.fetchone()
            self._conn.commit()
            if new_row:
                return {"id": new_row[0], "type_name": new_row[1], "metadata_schema": new_row[2]}
        return {}

    def get_or_create_document_by_filename(self, filename: str) -> Optional[int]:
        """
        Retrieves the ID of a document by its filename. If it doesn't exist,
        it creates a new record and returns the new ID.
        """
        print(f"DB -> Getting or creating document record for: '{filename}'")
        with self._conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE filename = %s;", (filename,))
            result = cur.fetchone()
            if result:
                print(f"DB -> Found existing document with ID: {result[0]}")
                return result[0]
            else:
                print(f"DB -> No record found. Creating new document record for '{filename}'.")
                insert_sql = "INSERT INTO documents (filename) VALUES (%s) RETURNING id;"
                cur.execute(insert_sql, (filename,))
                new_id = cur.fetchone()[0]
                self._conn.commit()
                print(f"DB -> Created new document with ID: {new_id}")
                return new_id
        return None

    def insert_metadata_value(self, document_id: int, meta_key: str, meta_value: str):
        """
        Inserts or updates a single metadata key-value pair for a given document.
        If the key already exists for the document, its value is updated.
        """
        sql = """
        INSERT INTO document_metadata_values (document_id, meta_key, meta_value)
        VALUES (%s, %s, %s)
        ON CONFLICT (document_id, meta_key)
        DO UPDATE SET meta_value = EXCLUDED.meta_value;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (document_id, meta_key, meta_value))
        self._conn.commit()

    def close(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            print("Database connection closed.")

    def get_dynamic_filter_context(self) -> Dict[str, List[str]]:
        """
        Fetches a dynamic summary of all unique, filterable metadata keys and
        a sample of their corresponding values from the database. This provides
        rich, real-time context to the LLM about what filters are possible.

        Returns:
            A dictionary where keys are the dynamically discovered metadata fields
            (e.g., 'company_name', 'document_type') and values are lists of
            unique values for that field.
        """
        print("DB -> Fetching DYNAMIC metadata summary for context...")
        context_summary = {}

        get_keys_sql = "SELECT DISTINCT meta_key FROM document_metadata_values;"

        with self._conn.cursor() as cur:
            cur.execute(get_keys_sql)
            keys = [row[0] for row in cur.fetchall()]

            for key in keys:
                get_values_sql = """
                SELECT DISTINCT meta_value
                FROM document_metadata_values
                WHERE meta_key = %s
                ORDER BY meta_value
                LIMIT 20;
                """
                cur.execute(get_values_sql, (key,))
                values = [row[0] for row in cur.fetchall()]
                context_summary[key] = values

        if 'fiscal_year' in context_summary:
            context_summary['fiscal_year'] = sorted(
                [y for y in context_summary['fiscal_year'] if y.isdigit()],
                key=int,
                reverse=True
            )

        print(f"DB -> Found dynamic context: {len(context_summary.keys())} unique filterable fields.")
        return context_summary

    def get_recent_documents_summary(self) -> List[Dict[str, Any]]:
        """
        Fetches a list of the 50 most recently processed documents and their
        associated metadata.
        """
        print("DB -> Fetching summary of recent documents...")
        document_summary = []
        sql = """
              SELECT d.id, d.filename, v.meta_key, v.meta_value
              FROM documents d
                       LEFT JOIN document_metadata_values v ON d.id = v.document_id
              WHERE d.id IN (SELECT id FROM documents ORDER BY processed_at DESC LIMIT 50)
              ORDER BY d.id, v.meta_key; \
              """
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

            docs_by_id = {}
            for doc_id, filename, meta_key, meta_value in rows:
                if doc_id not in docs_by_id:
                    docs_by_id[doc_id] = {"filename": filename, "metadata": {}}
                if meta_key:
                    docs_by_id[doc_id]["metadata"][meta_key] = meta_value

            document_summary = list(docs_by_id.values())

        print(f"DB -> Found {len(document_summary)} recent docs.")
        return document_summary

