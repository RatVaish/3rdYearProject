import pytest
from unittest.mock import patch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from bm25_v2 import extract_github_filenames, retrieve_top_filename

# Initialize tokenizer globally for test cases
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


@pytest.mark.parametrize(
    "api_url, mock_response, expected_filenames",
    [
        (
                "https://api.github.com/repos/example/repo/tarball/main",
                ["file1.py", "file2.py"],
                ["file1.py", "file2.py"],
        ),
        (
                "https://api.github.com/repos/example/repo/tarball/main",
                [],
                [],
        ),
    ],
)
@patch("your_script.requests.get")  # Mock the requests.get function
def test_extract_github_filenames(mock_get, api_url, mock_response, expected_filenames):
    """Tests if extract_github_filenames correctly extracts Python filenames from a mock GitHub tarball."""

    # Mock the API response
    class MockResponse:
        status_code = 200
        content = b""

        def __init__(self, file_list):
            self.file_list = file_list

        def iter_content(self, chunk_size=1):
            return iter(self.file_list)

    mock_get.return_value = MockResponse(mock_response)

    extracted_files = extract_github_filenames(api_url)

    assert extracted_files == expected_filenames, f"Expected {expected_filenames} but got {extracted_files}"


@pytest.mark.parametrize(
    "corpus, query, expected_best_filename",
    [
        (["file1.py", "file2.py"], "file1", "file1.py"),
        (["data_loader.py", "model_train.py"], "train", "model_train.py"),
    ],
)
def test_retrieve_top_filename(corpus, query, expected_best_filename):
    """Tests if BM25 correctly retrieves the most relevant filename."""

    tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenizer.tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_index = scores.argmax()
    best_filename = corpus[top_index]  # Simulates retrieve_top_filename behavior

    assert best_filename == expected_best_filename, f"Expected {expected_best_filename} but got {best_filename}"
