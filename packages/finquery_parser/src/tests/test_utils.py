from unittest.mock import MagicMock, patch

import pytest

from finquery_parser.utils import (clean_element_text,
                                   get_one_line_summary, TableSummarizationError, Context, strip_thinking_tags)


def test_clean_element_text_removes_urls():
    """Verify that standard URLs are stripped from the text."""
    text = "This is a text with a link http://example.com and another one https://google.com."
    expected_actual_output = "This is a text with a link and another one."
    assert clean_element_text(text).strip() == expected_actual_output


@pytest.mark.xfail(reason="The current regex in `clean_element_text` is too greedy and removes trailing punctuation.")
def test_clean_element_text_removes_sec_links_preserves_punctuation():
    """Verify that specific SEC filing links are removed, preserving punctuation."""
    text = "Find more at www.sec.gov/Archives/edgar/data/123/abc.htm."
    expected = "Find more at."
    assert clean_element_text(text).strip() == expected


def test_clean_element_text_consolidates_newlines():
    """Verify that multiple sequential newlines are reduced to one."""
    text = "Line 1\n\n\nLine 2"
    expected = "Line 1\nLine 2"
    assert clean_element_text(text) == expected


def test_clean_element_text_no_change():
    """Verify that clean text remains unchanged."""
    text = "This is a clean sentence with no junk."
    assert clean_element_text(text) == text

@pytest.fixture
def sample_context():
    """Provides a sample Context object for tests."""
    return Context(section_title="Consolidated Balance Sheets")

@patch('finquery_parser.utils.ChatPromptTemplate')
def test_get_one_line_summary_success(mock_prompt_template):
    """
    Test the successful path of get_one_line_summary by mocking the entire LLM chain.
    """
    mock_chain = MagicMock()
    mock_prompt_template.from_messages.return_value = mock_chain
    expected_summary = {"summary": "A summary of financial data."}
    mock_chain.__or__().__or__().__or__().invoke.return_value = expected_summary

    result = get_one_line_summary(table_text="<table>...</table>", section_title="Financial Highlights",
        parser=MagicMock(), llm=MagicMock())

    mock_chain.__or__().__or__().__or__().invoke.assert_called_once_with(
        {"table": "<table>...</table>", "section_title": "Financial Highlights"})
    assert result == expected_summary


def test_strip_thinking_tags_removes_tags():
    """Verify that the strip_thinking_tags function correctly removes the tags."""
    raw_text = "<think>This is some thinking.</think>This is the real content."
    expected = "This is the real content."
    assert strip_thinking_tags(raw_text) == expected


def test_get_one_line_summary_llm_failure():
    """
    Test that a TableSummarizationError is raised if the LLM chain fails.
    """
    with patch('finquery_parser.utils.ChatPromptTemplate.from_messages') as mock_from_messages:
        mock_chain = mock_from_messages.return_value.__or__().__or__().__or__()
        mock_chain.invoke.side_effect = Exception("LLM API is down")

        with pytest.raises(TableSummarizationError,
                           match="LLM call failed for table in section 'Financial Highlights'"):
            get_one_line_summary(table_text="<table>...</table>", section_title="Financial Highlights",
                parser=MagicMock(), llm=MagicMock())
