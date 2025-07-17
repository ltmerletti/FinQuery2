from unittest.mock import patch, MagicMock

import pytest

from finquery_app.database.delete_collection import delete_collection_and_folder
from finquery_app.manager import get_llm


@patch('finquery_app.database.manager.ChatOpenAI')
def test_get_llm_uses_defaults(mock_chat_openai):
    """Verify get_llm calls ChatOpenAI with default config values when none are provided."""
    get_llm()
    mock_chat_openai.assert_called_once()


@patch('finquery_app.database.manager.ChatOpenAI')
def test_get_llm_uses_provided_args(mock_chat_openai):
    """Verify get_llm uses provided arguments instead of defaults."""
    get_llm(base_url="http://test.url", api_key="test_key", model_name="test_model")
    mock_chat_openai.assert_called_once_with(model="test_model", api_key="test_key", base_url="http://test.url",
        temperature=0.1)


@patch('finquery_app.database.delete_collection.pathlib.Path.exists', return_value=True)
@patch('finquery_app.database.delete_collection.pathlib.Path.is_dir', return_value=True)
@patch('finquery_app.database.delete_collection.chromadb.PersistentClient')
def test_delete_collection_and_folder_success(mock_persistent_client, _mock_is_dir, _mock_exists):
    """
    Test that the delete function correctly calls the client's delete_collection method.
    """
    mock_client_instance = MagicMock()
    mock_persistent_client.return_value = mock_client_instance

    delete_collection_and_folder("my_test_collection", "/fake/db/path")

    mock_persistent_client.assert_called_once_with(path="/fake/db/path")
    mock_client_instance.delete_collection.assert_called_once_with(name="my_test_collection")


@patch('finquery_app.database.delete_collection.pathlib.Path.exists', return_value=True)
@patch('finquery_app.database.delete_collection.pathlib.Path.is_dir', return_value=True)
@patch('finquery_app.database.delete_collection.chromadb.PersistentClient')
def test_delete_collection_handles_non_existent(mock_persistent_client, _mock_is_dir, _mock_exists):
    """
    Test that the delete function handles a non-existent collection gracefully.
    """
    mock_client_instance = MagicMock()
    mock_client_instance.delete_collection.side_effect = ValueError("Collection not found")
    mock_persistent_client.return_value = mock_client_instance

    try:
        delete_collection_and_folder("non_existent_collection", "/fake/db/path")
    except Exception as e:
        pytest.fail(f"delete_collection_and_folder raised an unexpected exception: {e}")

    mock_client_instance.delete_collection.assert_called_once_with(name="non_existent_collection")