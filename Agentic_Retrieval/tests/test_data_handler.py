import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from Agentic_Retrieval.data_handler import DataHandler


# Sample data for testing purposes
sample_data = {
    'problem_statements': [
        "Given a list of numbers, find the sum of the squares.",
        "Implement a function that multiplies two matrices."
    ],
    'github_api_url': [
        "https://api.github.com/repos/user/repo/tarball/commit_hash"
    ],
    'patch_files': [
        "file1.py"
    ]
}

# Prepare a mock DataFrame to pass to the DataHandler class
df = pd.DataFrame(sample_data)


@pytest.fixture
def data_handler():
    """Fixture to create a DataHandler instance."""
    return DataHandler(db_path="cleaned_data.csv")


@pytest.fixture
def mock_data_handler():
    """Fixture for DataHandler with mock data."""
    return DataHandler(db_path=None)


def test_fetch_problem_statements(mock_data_handler):
    problem_statements = mock_data_handler.ProblemStatements(df).fetch()
    assert len(problem_statements) == 2
    assert problem_statements[0] == "Given a list of numbers, find the sum of the squares."


def test_fetch_problem_keywords(mock_data_handler):
    statement = "Given a list of numbers, find the sum of the squares."
    keywords = mock_data_handler.ProblemStatements.fetch_problem_keywords(statement)
    assert "numbers" in keywords
    assert "sum" in keywords
    assert "squares" in keywords


def test_extract_function_name(mock_data_handler):
    statement = "Define a function `sum_squares(numbers)` that returns the sum of squares."
    functions = mock_data_handler.ProblemStatements.extract_function_name(statement)
    assert "sum_squares" in functions


def test_preprocess_problem(mock_data_handler):
    statement = """Here is the problem description.
    ```python
    def add(a, b):
        return a + b
    ```"""
    processed = mock_data_handler.ProblemStatements.preprocess_problem(statement)
    assert "Here is the problem description" in processed
    assert "def add(a, b)" in processed


def test_fetch_code(mock_data_handler):
    # Mocking requests.get to return a sample tarball response
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"mock tarball content"
        mock_get.return_value = mock_response

        result = mock_data_handler.GithubAPIUrls.fetch_code("mock_url")
        assert isinstance(result, dict)  # The return value should be a dictionary


def test_fetch_filenames(mock_data_handler):
    # Mocking requests.get to return a sample tarball response
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"mock tarball content"
        mock_get.return_value = mock_response

        result = mock_data_handler.GithubAPIUrls.fetch_filenames("mock_url")
        assert isinstance(result, list)  # The return value should be a list


def test_fetch_patch_files(mock_data_handler):
    patch_files = mock_data_handler.PatchFiles(df).fetch()
    assert len(patch_files) == 1
    assert patch_files[0] == "file1.py"
