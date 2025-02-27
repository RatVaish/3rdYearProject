#-----Imports-----
import openai
import os
from data_handler import DataHandler
from search_tools import SearchTools


#-----Agent Class-----
class CodeRetrievalAgent:
    def __init__(self, db_path="cleaned_data.csv", use_llm=True):
        self.data_handler = DataHandler(db_path)
        self.problem_statements = self.data_handler.ProblemStatements(self.data_handler.df)
        self.github_urls = self.data_handler.GithubAPIUrls(self.data_handler.df)
        self.use_llm = use_llm

    def run(self, index=0):
        problem_statement = self.problem_statements.fetch()[index]
        github_api_url = self.github_urls.fetch()[index]

        keywords = self.problem_statements.fetch_problem_keywords(problem_statement)
        function_names = self.problem_statements.extract_function_name(problem_statement)

        file_contents = self.github_urls.fetch_code(github_api_url)
        filenames = list(file_contents.keys())

        regex_search = SearchTools.RegexRetriever()
        regex_search.index_code(file_contents)
        regex_match = regex_search.search("|".join(keywords))

        ast_search = SearchTools.ASTRetriever(file_contents)
        ast_results = {}
        for function_name in function_names:
            ast_results.update(ast_search.search(function_name))

        # Step 4: Decision Logic
        if regex_match:
            return regex_match
        elif ast_results:
            return list(ast_results.keys())[0]
        elif self.use_llm:
            return self.query_llm(problem_statement, filenames)
        else:
            return None

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
