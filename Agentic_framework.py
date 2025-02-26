#-----Imports-----
import pandas as pd
import os
import requests
import tarfile
import io
import spacy
import re
import torch
from transformers import AutoTokenizer, AutoModel


#-----Classes-----
class DataHandler:
    def __init__(self, db_path="cleaned_data.csv"):
        self.db_path = db_path
        if os.path.exists(self.db_path):
            self.df = pd.read_csv(self.db_path)
        else:
            raise ValueError("Database path does not exist")

    def fetch_code(self, github_api_url):

        '''
        Extract all project files from a GitHub repository tarball url at a specific
        commit hash. Storing the contents in a dictionary.

        :param github_api_url: (str) The  GitHub API url for the tarball of a
                                repository at a specific commit hash.
        :return              : (dict) A dictionary where keys are file paths and
                                values are the contents of the files.
        '''

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

    def fetch_filenames(self, github_api_url):

        '''
        Extract all project filenames from a GitHub repository tarball url at a specific
        commit hash. Storing the contents as a list.

        :param github_api_url: (str) The  GitHub API url for the tarball
                                of a project at a specific commit hash.
        :return              : (list) List of filenames in the repository.
        '''

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

    def fetch_problems(self):
        return self.df["problem_statements"]

    def fetch_problem_keywords(self, problem_statement):
        '''
        Extracts keywords (nouns, verbs, etc.) from a problem statement
        using spaCy's NLP capabilities and returns them as a list.

        :param problem_statement: (str) Problem description from SWE Bench
        :return                 : (list) List of keywords extracted from problem statement
        '''

        if not isinstance(problem_statement, str):
            return []

        doc = nlp(problem_statement)
        keywords = {token.text.lower() for token in doc if token.pos_ in ["NOUN"]}

        return list(keywords)

    def fetch_answers(self):
        return self.df["patch_files"]


class SearchTools:
    class EmbeddingHandler:
        def __init__(self, model_name="microsoft/codebert-base"):
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        def change_model(self, new_model):

            '''
            Sets a different code embedding model (Initially CodeBERT base).
            Allowing for different embeddings to evaluate.

            :param new_model: (str) The new code embedding model.
            :return         : (None) Updated code embedding model and tokenizer model.
            '''

            self.model_name = new_model
            self.tokenizer = AutoTokenizer.from_pretrained(new_model)
            self.model = AutoModel.from_config(new_model)

        def get_embeddings(self, text):

            '''
            Uses the embedding model (Initially CodeBERT base) to generate embeddings
            for a given text. This applies to any input text.

            :param text: (str) The input text that an embedding is generated for.
            :return    : (numpy.ndarray) Vector representation of the input text.
            '''

            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**tokens)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    class RegexRetriever:
        def __init__(self):
            self.indexed_files = {}

        def index_code(self, file_contents):

            '''
            Indexes the filenames by embedding their contents using the specified
            embedding model.

            :param file_contents: (dict) The dictionary where keys are filenames and
                                  values are the contents of the files.
            :return             : (None) The files are indexed with their raw contents.
            '''

            self.indexed_files = file_contents

        def search(self, query):

            '''
            Searches indexed filenames for matches using regular expression.

            :param query: (str) The regular expression query to search for
            :return     : (str or None) The most relevant filename or None if no matches
            '''

            matches = {filename: len(re.findall(query, filename)) for filename in self.indexed_files}

            # Filter out filenames with no matches
            matches = {filename: count for filename, count in matches.items() if count > 0}

            if matches:
                # Return the filename with the highest match count
                return max(matches, key=matches.get)
            else:
                return None


class TestingResults:
    pass


#-----Body-----
data_handler = DataHandler()

#--Regex Search--
nlp = spacy.load("en_core_web_sm")
regex_retriever = SearchTools.RegexRetriever()

for idx, row in data_handler.df.iterrows():
    github_api_url = row["github_api_url"]

    filenames = data_handler.fetch_filenames(github_api_url)
    regex_retriever.index_code(filenames)

    keywords = data_handler.fetch_problem_keywords(row["problem_statement"])

    if not keywords:
        print(f"No keywords extracted for problem {idx}, skipping search.")
        continue

    regex_pattern = r"\b(" + "|".join(map(re.escape, keywords)) + r")\b"
    regex_results = regex_retriever.search(regex_pattern)

    if regex_results:
        print(f"Regex search results for problem {idx}: {regex_results}")
    else:
        print(f"No matches found for problem {idx}.")
