import pathlib
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from finquery_app.ingestion.pipeline import run_ingestion_process


@patch('finquery_app.ingestion.pipeline.shutil.move')
@patch('finquery_app.ingestion.pipeline.index')
@patch('finquery_app.ingestion.pipeline.CustomPDFLoader')
@patch('finquery_app.ingestion.pipeline.get_file_paths')
def test_run_ingestion_process_full_flow(mock_get_paths, mock_loader, mock_index, mock_move):
    """
    Test the full ingestion pipeline logic without touching the filesystem or external services.
    Verifies that files are found, loaded, indexed, and moved correctly.
    """
    dummy_path = pathlib.Path('/fake/dir/aapl-10k.pdf')
    mock_get_paths.return_value = [dummy_path]

    mock_docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    mock_loader.return_value.load.return_value = mock_docs

    mock_vector_store = MagicMock()
    mock_record_manager = MagicMock()
    mock_llm = MagicMock()

    run_ingestion_process(mock_vector_store, mock_record_manager, mock_llm)

    mock_get_paths.assert_called_once()

    mock_loader.assert_called_once_with(str(dummy_path), mock_llm)
    mock_loader.return_value.load.assert_called_once()

    mock_index.assert_called_once_with(mock_docs, mock_record_manager, mock_vector_store, cleanup="incremental",
        source_id_key="source", batch_size=64)

    mock_move.assert_called_once()
    assert mock_move.call_args[0][0] == dummy_path


@patch('finquery_app.ingestion.pipeline.get_file_paths')
def test_run_ingestion_process_no_files(mock_get_paths):
    """
    Test that the ingestion process exits gracefully when no new files are found.
    """
    mock_get_paths.return_value = []
    mock_vector_store = MagicMock()
    mock_record_manager = MagicMock()
    mock_llm = MagicMock()

    with patch('finquery_app.ingestion.pipeline.index') as mock_index:
        run_ingestion_process(mock_vector_store, mock_record_manager, mock_llm)

        mock_index.assert_not_called()
