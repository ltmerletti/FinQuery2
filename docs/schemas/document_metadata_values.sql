CREATE TABLE IF NOT EXISTS document_metadata_values (
    id SERIAL PRIMARY KEY,
    document_type_id INTEGER NOT NULL REFERENCES document_types(id) ON DELETE CASCADE,
    metadata_key VARCHAR(128) NOT NULL,
    metadata_value VARCHAR(256) NOT NULL,
    UNIQUE (document_type_id, metadata_key, metadata_value)
);
