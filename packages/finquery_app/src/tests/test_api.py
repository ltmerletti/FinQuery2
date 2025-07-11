import json
from io import BytesIO
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from finquery_app.api.main import app


@pytest.fixture
def client():
    """Create and configure a new app instance for each test."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test the health check endpoint for a simple 200 OK response."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert "FinQuery API is running" in data['message']


def test_upload_file_success(client):
    """Test successful file upload with a mock PDF."""
    data = {'file': (BytesIO(b'fake pdf content'), 'test_report.pdf')}
    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    assert response.status_code == 201
    assert b"File 'test_report.pdf' uploaded successfully" in response.data


def test_upload_file_no_file_part(client):
    """Test the upload endpoint when no 'file' part is in the request."""
    response = client.post('/api/upload', content_type='multipart/form-data', data={})
    assert response.status_code == 400
    assert b"No file part in the request" in response.data


def test_upload_file_invalid_type(client):
    """Test the upload endpoint with a non-PDF file, which should be rejected."""
    data = {'file': (BytesIO(b'some text'), 'test.txt')}
    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    assert response.status_code == 400
    assert b"Invalid file type" in response.data


@patch('finquery_app.api.main.threading.Thread')
def test_trigger_ingestion_endpoint(mock_thread, client):
    """Test that the /api/ingest endpoint correctly starts a background thread."""
    response = client.post('/api/ingest')
    assert response.status_code == 202
    mock_thread.assert_called_once()
    mock_thread.return_value.start.assert_called_once()
    assert b"Ingestion process started" in response.data


@patch('finquery_app.api.main.execute_query')
def test_query_endpoint(mock_execute_query, client):
    """Test the /api/query endpoint, mocking the query execution logic."""
    mock_doc = Document(page_content="Apple's net income was $97.0 billion.",
                        metadata={"source": "aapl-10k.pdf", "page": 50})
    mock_execute_query.return_value = [mock_doc]

    response = client.post('/api/query', json={'query_text': 'net income'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['query'] == 'net income'
    assert len(data['results']) == 1
    assert data['results'][0]['content'] == "Apple's net income was $97.0 billion."
    assert data['results'][0]['metadata']['page'] == 50


@patch('finquery_app.api.main.retrieval_chain')
def test_ask_question_endpoint(mock_retrieval_chain, client):
    """Test the /api/question endpoint, mocking the retrieval chain."""
    mock_retrieval_chain.invoke.return_value = "The answer is 42. (Source: guide.pdf, Page: 1)"

    response = client.post('/api/question', json={'query_text': 'what is the meaning of life?'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['answer'] == "The answer is 42. (Source: guide.pdf, Page: 1)"
    mock_retrieval_chain.invoke.assert_called_once_with('what is the meaning of life?')


@patch('finquery_app.api.main.delete_collection_and_folder')
def test_delete_collection_endpoint(mock_delete_func, client):
    """Test the admin endpoint for deleting a collection, ensuring the backend function is called."""
    response = client.delete('/api/admin/collection')
    assert response.status_code == 200
    mock_delete_func.assert_called_once()
    assert b"Collection 'financial_documents' and its data have been deleted" in response.data
