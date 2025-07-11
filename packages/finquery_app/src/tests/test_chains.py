from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

from finquery_app.chains.answer_chain import create_rag_chain, format_docs_with_metadata


def test_format_docs_with_metadata_formats_correctly():
    """
    Verify that documents are formatted into a single string with correct metadata.
    """
    docs = [Document(page_content="[CONTEXT]\nDetails about revenue.\n[CONTENT]\nNet sales for iPhone were $200B.",
        metadata={"source": "aapl-10k.pdf", "page": 50}),
        Document(page_content="[CONTEXT]\nInfo on R&D.\n[CONTENT]\nResearch and development spending was $30B.",
            metadata={"source": "aapl-10k.pdf", "page": 25})]

    expected_output = ("Source: aapl-10k.pdf, Page: 50\n"
                       "Content:\n"
                       "Net sales for iPhone were $200B.\n\n"
                       "---\n\n"
                       "Source: aapl-10k.pdf, Page: 25\n"
                       "Content:\n"
                       "Research and development spending was $30B.")

    formatted_string = format_docs_with_metadata(docs)
    assert formatted_string == expected_output


def test_format_docs_with_no_content_tag():
    """
    Test that formatting still works if the [CONTENT] tag is missing.
    """
    docs = [Document(page_content="Full content without tag.", metadata={"source": "a.pdf", "page": 1})]
    expected = "Source: a.pdf, Page: 1\nContent:\nFull content without tag."
    assert format_docs_with_metadata(docs) == expected


def test_format_docs_with_empty_list():
    """
    Test that the function returns a specific message for an empty document list.
    """
    assert format_docs_with_metadata([]) == "No documents found."


@patch('finquery_app.chains.answer_chain.ChatOpenAI')
@patch('finquery_app.chains.answer_chain.ContextualCompressionRetriever')
@patch('finquery_app.chains.answer_chain.CrossEncoderReranker')
@patch('finquery_app.chains.answer_chain.QwenReranker')
def test_create_rag_chain_structure(mock_qwen, mock_reranker, mock_retriever, mock_llm):
    """
    Verify that `create_rag_chain` initializes and assembles all components correctly.
    This is a structural test to ensure the chain is built as designed.
    """
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_retriever_instance = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever_instance

    rag_chain = create_rag_chain(mock_vector_store)

    mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={'k': 10})

    mock_qwen.assert_called_once()
    mock_reranker.assert_called_once_with(model=mock_qwen.return_value, top_n=4)

    mock_retriever.assert_called_once_with(base_compressor=mock_reranker.return_value,
        base_retriever=mock_retriever_instance)

    mock_llm.assert_called_once()

    assert isinstance(rag_chain, Runnable)