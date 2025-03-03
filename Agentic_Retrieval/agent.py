#-----Imports-----
import openai
import os
from .data_handler import DataHandler
from .search_tools import SearchTools
from typing import Optional, List


#-----Agent Class-----
class CodeRetrievalAgent:
    def __init__(self, db_path: str = "cleaned_data.csv", use_llm: bool = True):

        self.data_handler = DataHandler(db_path)
        self.problem_statement = self.data_handler.ProblemStatements.fetch
        self.github_api_urls = self.data_handler.GithubAPIUrls.fetch
        self.use_llm = use_llm
        self.preprocessed_problem = self.data_handler.ProblemStatements.preprocess_problem(self.problem_statement)
        self.code_snippets = self.data_handler.GithubAPIUrls.fetch_code(self.github_api_urls)

        self.search_tools = {
            'embedding': None,
            'regex': None,
            'ast': None,
            'symbolic': None,
            'call_graph': None,
            'docstring': None,
            'import': None,
            'heuristic': None
        }

    def get_search_tools(self, tool_name: str, file_contents: Optional[dict] = None):

        if self.search_tools[tool_name] is None:
            if tool_name == 'embedding':
                self.search_tools['embedding'] = SearchTools.EmbeddingIndex()
            elif tool_name == 'regex':
                self.search_tools['regex'] = SearchTools.RegexRetriever()
            elif tool_name == 'ast':
                self.search_tools['ast'] = SearchTools.ASTRetriever(file_contents)
            elif tool_name == 'symbolic':
                self.search_tools['symbolic'] = SearchTools.SymbolicRetriever(file_contents)
            elif tool_name == 'call_graph':
                self.search_tools['call_graph'] = SearchTools.CallGraphAnalyser()
            elif tool_name == 'docstring':
                self.search_tools['docstring'] = SearchTools.DocstringRetriever()
            elif tool_name == 'import':
                self.search_tools['import'] = SearchTools.ImportSearch()
            elif tool_name == 'heuristic':
                self.search_tools['heuristic'] = SearchTools.HeuristicScorer()
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        return self.search_tools[tool_name]

    def query_llm(self, prompt: str) -> str:

        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are an AI assistant for code retrieval."},
                      {"role": "user", "content": prompt}],
            max_tokens=50
        )
        return response["completions"][0]["message"]["content"].strip()

    def decide_best_search(self) -> str:

        prompt = f"""
                You are an intelligent code retrieval assistant. Given the following problem statement and code snippets, determine the best search method to find the relevant file.

                Problem Statement: {self.preprocessed_problem}
                Code Snippets: {self.code_snippets}

                Available search methods: embedding, regex, ast, symbolic, call_graph, docstring, import, heuristic.

                Respond with only the best search method name.
                """
        return self.query_llm(prompt)

    def refine_search(self, search_method: str, results: List[str]) -> str:

        if len(results) == 1:
            return results[0]  # Single filename found

        prompt = f"""
                You are an intelligent code retrieval assistant refining search results. 

                Problem Statement: {self.preprocessed_problem}
                Code Snippets: {self.code_snippets}
                Initial Search Method: {search_method}
                Initial Results: {results}

                Suggest the most relevant filename or an alternative search method.
                """
        return self.query_llm(prompt)

    def run(self, index: int) -> Optional[str]:
        code_snippets = self.code_snippets[index]

        best_search_method = self.decide_best_search()

        search_tool = self.get_search_tools(best_search_method, file_contents=code_snippets)
        results = search_tool.search(code_snippets)

        refined_result = self.refine_search(best_search_method, results)

        return refined_result
