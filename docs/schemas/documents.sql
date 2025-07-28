CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(512) NOT NULL UNIQUE,
                document_type_id INTEGER REFERENCES document_types(id) ON DELETE SET NULL,
                processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );