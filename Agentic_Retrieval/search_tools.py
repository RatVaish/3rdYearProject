from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import re
import ast
import networkx as nx
from data_handler import DataHandler


class SearchTools:
    class EmbeddingIndex:
        def __init__(self, model_name="microsoft/codebert-base"):
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        def change_model(self, new_model):

            """
            Sets a different code embedding model (Initially CodeBERT base).
            Allowing for different embeddings to evaluate.

            :param new_model: (str) The new code embedding model.
            :return         : (None) Updated code embedding model and tokenizer model.
            """

            self.model_name = new_model
            self.tokenizer = AutoTokenizer.from_pretrained(new_model)
            self.model = AutoModel.from_config(new_model)

        def get_embeddings(self, text):

            """
            Uses the embedding model (Initially CodeBERT base) to generate embeddings
            for a given text. This applies to any input text.

            :param text: (str) The input text that an embedding is generated for.
            :return    : (numpy.ndarray) Vector representation of the input text.
            """

            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        def index_codebase(self, file_contents):

            """
            Indexes all code files in FAISS for fast retrieval.

            :param file_contents: (Dict) Extracted GitHub files {filename: file_contents}
            :return             : (numpy.ndarray) Vector representation of the input codebase.
            """

            self.file_map = {i: filename for i, filename in enumerate(file_contents.keys())}
            embeddings = np.array([self.get_embeddings(content) for content in file_contents.values()], dtype=np.float32)

            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)

        def search(self, query, top_k=5):

            """
            Searches the indexed codebase for the most relevant files using a query
            extracted from the SWE Bench problem statement.

            :param query: (str) Preprocessed problem statement from SWE Bench.
            :param top_k: (int) Number of results to return.
            :return     : (list) Tok-k filenames.
            """

            if self.index is None:
                raise ValueError('FAISS index has not been built. Call build_index() first.')

            query_embedding = self.get_embeddings(query).reshape(1, -1).astype("float32")
            distances, indices = self.index.search(query_embedding, top_k)

            return [self.file_map[i] for i in indices[0] if i != -1]  # Return filenames

    class RegexRetriever:
        def __init__(self):
            self.indexed_files = {}

        def index_code(self, file_contents):

            """
            Indexes the filenames by embedding their contents using the specified
            embedding model.

            :param file_contents: (dict) The dictionary where keys are filenames and
                                  values are the contents of the files.
            :return             : (None) The files are indexed with their raw contents.
            """

            self.indexed_files = file_contents

        def search(self, query):

            """
            Searches indexed filenames for matches using regular expression.

            :param query: (str) The regular expression query to search for
            :return     : (str or None) The most relevant filename or None if no matches
            """

            matches = {filename: len(re.findall(query, filename)) for filename in self.indexed_files}

            # Filter out filenames with no matches
            matches = {filename: count for filename, count in matches.items() if count > 0}

            if matches:
                # Return the filename with the highest match count
                return max(matches, key=matches.get)
            else:
                return None

    class ASTRetriever:
        def __init__(self, file_contents):
            self.file_contents = file_contents

        def search(self, function_name):

            """
            Searches for a specific function defined in indexed file contents.
            Iterates through file contents and looks for function definitions
            matching the provided function name.

            :param function_name: (str) The function name to search for.
            :return             : (dict) A dictionary where keys are filenames and
                                  values are the contents of the files.
            """

            results = {}
            for filename, content in self.file_contents.items():
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if node.name == function_name:
                                results[filename] = content
                                break
                except SyntaxError:
                    continue
            return results

    class SymbolicRetriever:
        def __init__(self, symbol_index):
            self.symbol_index = symbol_index #Dict {filename: {"functions": [...], "classes": [...]}}

        def search(self, problem_statement):

            """
            Searches for function and class names from the problem statement in the
            codebase.

            :param problem_statement: (str) The problem statement to search from.
            :return                 : (Dict) Filenames where symbols are located.
            """

            query_symbols = DataHandler.ProblemStatements.extract_function_name(problem_statement)
            matches = {}

            for filename, symbols in self.symbol_index.items():
                if any(query in sym_list for query in query_symbols for sym_list in symbols.values()):
                    matches[filename] = symbols

            return matches

    class CallGraphAnalyser:
        def __init__(self):
            self.call_graph = nx.DiGraph()
            self. function_map = {}

        def build_call_graph(self, file_contents):

            """
            Parses the retrieved GitHub files and builds a function call graph
            for later use.

            :param file_contents: (Dict) Input files extracted from GitHub
                                  {filename: file_contents}.
            :return             : (None) Creates function call graph.
            """

            for filename, content in file_contents.items():
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        self.function_map[node.name] = filename
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        self.call_graph.add_edge(node.func.id, filename)

        def search(self, function_name):

            """
            Finds where a function is defined in the call graph and where it is
            used in the retrieved files.

            :param function_name: (str) The function name to search for.
            :return             : (set) Filenames where function is used.
            """

            if function_name in self.function_map:
                return{self.function_map[function_name]} | set(self.call_graph.predecessors(function_name))
            else:
                return set()

    class DocstringRetriever:
        def __init__(self):
            self.file_map = {}

        def extract_docstrings(self, file_contents):

            """
            Extracts docstrings and comments from the retrieved GitHub files.

            :param file_contents:
            :return             :
            """

            docstring_patten = r'"""(.*?)"""|\'\'\'(.*?)\'\'\'|#(.*?)\n'
            matches = re.findall(docstring_patten, file_contents, re.DOTALL)
            return " ".join(" ".join(m).strip() for m in matches)

        def index_docstring(self, file_contents):

            """
            Indexes the extracted docstrings and comments from the GitHub code
            files.

            :param file_contents:
            :return             :
            """

            self.file_map = {filename: self.extract_docstrings(content) for filename, content in file_contents.items()}

        def search(self, query):
            """
            Searches for relevant terms inside docstrings and comments against the
            problem statement query.

            :param query: (str) Preprocessed problem statement to search with.
            :return     : (set) Filenames where query appears in docstrings/comments.
            """

            return {filename for filename, docs in self.file_map.items() if query in docs}

    class ImportSearch:
        def __init__(self):
            self.import_map = {}

        def index_imports(self, file_contents):

            """

            :param file_contents:
            :return:
            """

            for filename, content in file_contents.items():
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.import_map.setdefault(alias.name, []).append(filename)
                    elif isinstance(node, ast.ImportFrom):
                        self.import_map.setdefault(node.module, []).append(filename)

        def search(self, module_name):

            """

            :param module_name:
            :return:
            """

            return self.import_map.get(module_name,[])

    class HeuristicScorer:
        def __init__(self, weights=None):
            self.weights = weights or {
                "symbolic": 1.5,
                "call_graph": 1.2,
                "docstring": 1.0,
                "import": 0.8,
                "embedding": 2.0,
                "tfidf": 1.5
            }

        def rank_files(self, results):

            """

            :param results:
            :return:
            """

            scores = {}
            for method, files in results.items():
                weight = self.weights.get(method, 1.0)
                for file in files:
                    scores[file] = scores.get(file, 0) + weight
            return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
