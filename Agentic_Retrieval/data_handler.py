#-----Imports-----
import os
import pandas as pd
import re
import requests
import tarfile
import io
import spacy


#-----Functions-----
class DataHandler:
    def __init__(self, db_path="cleaned_data.csv"):
        self.db_path = db_path
        if os.path.exists(self.db_path):
            self.df = pd.read_csv(self.db_path)
        else:
            raise ValueError("Database path does not exist")

    class ProblemStatements:
        def __init__(self, df):
            self.df = df

        def fetch(self):
            return self.df["problem_statements"]

        @staticmethod
        def fetch_problem_keywords(problem_statement):
            """
            Extracts keywords (nouns, verbs, etc.) from a problem statement
            using spaCy's NLP capabilities and returns them as a list.

            :param problem_statement: (str) Problem description from SWE Bench
            :return                 : (list) List of keywords extracted from problem statement
            """

            if not isinstance(problem_statement, str):
                return []

            doc = nlp(problem_statement)
            keywords = {token.text.lower() for token in doc if token.pos_ in ["NOUN"]}

            return list(keywords)

        @staticmethod
        def extract_function_name(problem_statement):

            """
            Extracts function names form the problem statement using python style
            function calls and class names.

            :param problem_statement: (str) Problem description from SWE Bench
            :return                 : (set) Extracted function and class names.
            """

            function_names = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', problem_statement)
            class_names = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', problem_statement)

            return set(function_names + class_names)

        @staticmethod
        def preprocess_problem(problem_statement):

            """
            Extracts the useful parts of the problem statement keeping descriptions
            and code snippets, while removing examples and formatting.

            :param problem_statement: (str) Problem description from SWE Bench
            :return                 : (str) Processed problem statement
            """

            description = re.split(r'```python', problem_statement, maxsplit=1)[0].strip()
            code_snippets = re.findall(r'```python\n(.*?)\n```', problem_statement, re.DOTALL)
            code = "\n".join(code_snippets).strip()

            processed_text = f"{description}\n\n{code}"

            return processed_text

    class GithubAPIUrls:
        def __init__(self, df):
            self.df = df

        def fetch(self):
            return self.df["github_api_url"]

        @staticmethod
        def fetch_code(github_api_url):

            """
            Extract all project files from a GitHub repository tarball url at a specific
            commit hash. Storing the contents in a dictionary.

            :param github_api_url: (str) The  GitHub API url for the tarball of a
                                    repository at a specific commit hash.
            :return              : (dict) A dictionary where keys are file paths and
                                    values are the contents of the files.
            """

            # Setting up my PAT
            GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
            HEADERS = {
                'Authorization': f"token {GITHUB_TOKEN}",
                "User-Agent": "python-requests"
            }

            response = requests.get(github_api_url, headers=HEADERS, stream=True)

            if response.status_code == 200:
                files_content = {}
                with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            try:
                                file_content = tar.extractfile(member).read().decode("utf-8")
                                files_content[member.name] = file_content
                            except UnicodeDecodeError:
                                continue
                return files_content
            else:
                print(f"Failed to retrieve the tarball, Status code: {response.status_code}")
                return {}

        @staticmethod
        def fetch_filenames(github_api_url):

            """
            Extract all project filenames from a GitHub repository tarball url at a specific
            commit hash. Storing the contents as a list.

            :param github_api_url: (str) The  GitHub API url for the tarball
                                    of a project at a specific commit hash.
            :return              : (list) List of filenames in the repository.
            """

            # Setting up my PAT
            GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
            HEADERS = {
                'Authorization': f"token {GITHUB_TOKEN}",
                "User-Agent": "python-requests"
            }

            response = requests.get(github_api_url, headers=HEADERS, stream=True)

            if response.status_code == 200:
                filenames = []
                with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            filenames.append(member.name)
                return filenames
            else:
                print(f"Failed to retrieve the tarball, Status code: {response.status_code}")
                return []

    class PatchFiles:
        def __init__(self, df):
            self.df = df

        def fetch(self):
            return self.df["patch_files"]


nlp = spacy.load("en_core_web_sm")
