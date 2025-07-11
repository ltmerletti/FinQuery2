from unittest.mock import MagicMock

from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore

from finquery_app.querying.query import execute_query, get_rag_test_questions


def test_get_rag_test_questions_returns_list_of_strings():
    """
    Verify that get_rag_test_questions returns a list containing only strings.
    """
    questions = get_rag_test_questions()
    assert isinstance(questions, list)
    assert len(questions) > 0
    assert all(isinstance(q, str) for q in questions)


def test_execute_query():
    """
    Test the execute_query function to ensure it correctly configures and
    invokes the retriever.
    """
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_retriever_instance = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever_instance

    query_text = "What was the revenue?"
    num_to_fetch = 15
    mock_config = MagicMock(spec=RunnableConfig)

    execute_query(query_text, mock_vector_store, num_to_fetch, mock_config)

    mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={'k': num_to_fetch})

    mock_retriever_instance.invoke.assert_called_once_with(query_text, config=mock_config)


def test_execute_query_with_empty_query_text():
    """
    Test that execute_query returns None if the query text is empty.
    """
    mock_vector_store = MagicMock(spec=VectorStore)
    result = execute_query("", mock_vector_store, 10, MagicMock())
    assert result is None
