import pytest
import numpy as np
from unittest.mock import patch
from RAG_traditional import extract_github_files, get_embedding, similarity_search


@pytest.mark.parametrize("mock_response, expected_output", [
    ({'file1.py': "print('Hello World')"}, {'file1.py': "print('Hello World')"}),
    ({}, {})  # Tests empty response
])
@patch("your_rag_module.requests.get")
def test_extract_github_files(mock_get, mock_response, expected_output):
    """Mocks GitHub API response and tests extraction."""

    class MockResponse:
        def __init__(self, content, status_code):
            self.content = content
            self.status_code = status_code

    tar_content = b""

    mock_get.return_value = MockResponse(tar_content, 200)

    assert extract_github_files("https://api.github.com/repos/user/repo/tarball") == expected_output


@pytest.mark.parametrize("input_text", [
    ("def foo(): return 42"),
    ("class Example: pass"),
    ("")
])
def test_get_embedding(input_text):
    """Tests whether embeddings are generated correctly."""

    embedding = get_embedding(input_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[1] > 0


@pytest.mark.parametrize("problem_statement, github_files, expected_file", [
    ("Find the function returning 42", {"file1.py": "def foo(): return 42"}, "file1.py"),
    ("Define an empty class", {"file2.py": "class Example: pass"}, "file2.py")
])
def test_similarity_search(problem_statement, github_files, expected_file):
    """Tests similarity search using mock embeddings."""

    assert similarity_search(problem_statement, github_files) == expected_file
