import pytest
import numpy as np
from Agentic_Retrieval.search_tools import SearchTools

# Mock data for testing
mock_code_files = {
    "file1.py": "def foo(): pass\n\nclass Bar: pass",
    "file2.py": "def foo(): pass\n\nclass Baz: pass",
    "file3.py": "def qux(): pass\n\nclass Bar: pass"
}
mock_problem_statement = "Search for the function foo and class Bar"


@pytest.fixture
def setup_search_tools():
    """Fixture to set up SearchTools classes"""
    return {
        "embedding": SearchTools.EmbeddingIndex(),
        "regex": SearchTools.RegexRetriever(),
        "ast": SearchTools.ASTRetriever(mock_code_files),
        "symbolic": SearchTools.SymbolicRetriever({
            "file1.py": {"functions": ["foo"], "classes": ["Bar"]},
            "file2.py": {"functions": ["foo"], "classes": ["Baz"]},
            "file3.py": {"functions": ["qux"], "classes": ["Bar"]}
        }),
        "call_graph": SearchTools.CallGraphAnalyser(),
        "docstring": SearchTools.DocstringRetriever(),
        "import": SearchTools.ImportSearch()
    }


def test_embedding_index(setup_search_tools):
    # Test get_embeddings
    embedding_index = setup_search_tools["embedding"]
    text = "def foo(): pass"
    embedding = embedding_index.get_embeddings(text)
    assert isinstance(embedding, np.ndarray), "get_embeddings should return a numpy ndarray"


def test_regex_retriever(setup_search_tools):
    # Test search function for regex retriever
    regex_retriever = setup_search_tools["regex"]
    regex_retriever.index_code(mock_code_files)
    result = regex_retriever.search(r"foo")
    assert result == "file1.py", "Regex retriever should find file1.py with function foo"


def test_ast_retriever(setup_search_tools):
    # Test ASTRetriever search for function name
    ast_retriever = setup_search_tools["ast"]
    results = ast_retriever.search("foo")
    assert "file1.py" in results, "AST retriever should find foo in file1.py"
    assert "file2.py" in results, "AST retriever should find foo in file2.py"


def test_symbolic_retriever(setup_search_tools):
    # Test SymbolicRetriever search for function and class names
    symbolic_retriever = setup_search_tools["symbolic"]
    results = symbolic_retriever.search(mock_problem_statement)
    assert "file1.py" in results, "Symbolic retriever should find file1.py with Bar and foo"
    assert "file2.py" in results, "Symbolic retriever should find file2.py with Bar and foo"


def test_call_graph_analyser(setup_search_tools):
    # Test CallGraphAnalyser functionality
    call_graph = setup_search_tools["call_graph"]
    call_graph.build_call_graph(mock_code_files)
    results = call_graph.search("foo")
    assert "file1.py" in results, "Call graph should find foo in file1.py"


def test_docstring_retriever(setup_search_tools):
    # Test DocstringRetriever extraction
    docstring_retriever = setup_search_tools["docstring"]
    docstring_retriever.index_docstring(mock_code_files)
    result = docstring_retriever.search("function foo")
    assert "file1.py" in result, "Docstring retriever should find foo in file1.py"


def test_import_search(setup_search_tools):
    # Test ImportSearch functionality
    import_search = setup_search_tools["import"]
    import_search.index_imports(mock_code_files)
    result = import_search.search("os")
    assert result == [], "Import search should return an empty list as 'os' is not imported"


def test_heuristic_rank_files(setup_search_tools):
    # Test HeuristicScorer functionality
    heuristic_scorer = SearchTools.HeuristicScorer()
    results = {
        "symbolic": ["file1.py", "file2.py"],
        "call_graph": ["file2.py", "file3.py"]
    }
    ranked_files = heuristic_scorer.rank_files(results)
    assert ranked_files["file2.py"] > ranked_files["file1.py"], "Heuristic scorer should rank file2.py higher"
