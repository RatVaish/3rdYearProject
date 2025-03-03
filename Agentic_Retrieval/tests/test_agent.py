import pytest
from Agentic_Retrieval.agent import CodeRetrievalAgent
from unittest.mock import patch


def mock_query_llm(prompt: str) -> str:
    """Mock function to simulate LLM responses."""
    if "Initial Results" in prompt:
        return "alternative_search_method" if "multiple results" in prompt else "relevant_file.py"
    return "regex"  # Default mock response for decide_best_search


@pytest.fixture
def agent():
    return CodeRetrievalAgent(db_path="test_data.csv", use_llm=True)


def test_decide_best_search(agent):
    with patch.object(agent, 'query_llm', side_effect=mock_query_llm):
        search_method = agent.decide_best_search()
        assert search_method in ["embedding", "regex", "ast", "symbolic", "call_graph", "docstring", "import",
                                 "heuristic"]


def test_refine_search_returns_filename(agent):
    with patch.object(agent, 'query_llm', side_effect=mock_query_llm):
        result = agent.refine_search("regex", ["relevant_file.py"])
        assert result == "relevant_file.py"


def test_refine_search_returns_search_method(agent):
    with patch.object(agent, 'query_llm', side_effect=mock_query_llm):
        result = agent.refine_search("regex", ["file1.py", "file2.py"])  # Simulating multiple results
        assert result == "alternative_search_method"


def test_run(agent):
    with patch.object(agent, 'query_llm', side_effect=mock_query_llm):
        with patch.object(agent, 'get_search_tools') as mock_get_tool:
            mock_tool = mock_get_tool.return_value
            mock_tool.search.return_value = ["file1.py", "file2.py"]

            result = agent.run(0)
            assert result == "alternative_search_method"  # Expected refinement

            mock_tool.search.return_value = ["relevant_file.py"]
            result = agent.run(0)
            assert result == "relevant_file.py"
