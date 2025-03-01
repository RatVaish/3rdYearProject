#-----Imports-----
import openai
import os
from data_handler import DataHandler
from search_tools import SearchTools
from typing import Optional, List


#-----Agent Class-----
class CodeRetrievalAgent:
    def __init__(self, db_path: str = "cleaned_data.csv", use_llm: bool = True,):

        """
        Initialize the agent with the given database and search tools.

        :param db_path: Path to the data CSV file.
        :param use_llm: Flag to decide whether to use LLM for missing results.
        """

        self.data_handler = DataHandler(db_path)
        self.problem_statements = self.data_handler.ProblemStatements(self.data_handler.df)
        self.github_urls = self.data_handler.GithubAPIUrls(self.data_handler.df)
        self.use_llm = use_llm

        # Initialize search tools if not passed
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

        """
        Loads search tools as needed.

        :param tool_name    : (str) The name of the search tool to load.
        :param file_contents: (Optional[dict]) Required for tools like AST that need extra
                               information to load.
        :return             : (None) Initialised search tools.
        """

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

    def run(self, index: int = 0) -> Optional[str]:

        """

        :param index:
        :return:
        """

        problem_statement = self.problem_statements.fetch()[index]
        github_api_url = self.github_urls.fetch()[index]

        keywords = self.problem_statements.fetch_problem_keywords(problem_statement)
        function_names = self.problem_statements.extract_function_name(problem_statement)

        # Fetch GitHub Code
        file_contents = self.github_urls.fetch_code(github_api_url)
        filenames = list(file_contents.keys())

        # Embedding Search (FAISS)
        embedding_search = self.get_search_tools('embedding')
        embedding_search.index_code(file_contents)
        embedding_results = embedding_search.search("|".join(keywords), top_k=5)

        # Regex Search
        regex_search = self.get_search_tools('regex')
        regex_search.index_code(file_contents)
        regex_match = regex_search.search("|".join(keywords))

        # AST Search
        ast_search = self.get_search_tools('ast', file_contents)
        ast_results = {}
        for function_name in function_names:
            ast_results.update(ast_search.search(function_name))





    def query_llm(self, problem_statement, filenames):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        prompt = f"""
        You are a helpful AI agent tasked with identifying which filename is most relevant
        to the following problem statement.

        Problem Statement:
        {problem_statement}

        Here are the available filenames:
        {filenames}

        Return only the most relevant filename.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are an AI assistant."},
                      {"role": "user", "content": prompt}]
        )

        return response["choices"][0]["message"]["content"].strip()
