#-----Imports-----
import pandas as pd
import os
import requests
import tarfile
import io
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

    def fetch_problems(self):
        return self.df["problem_statements"]

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
            Indexes the code files by embedding their contents using the specified
            embedding model.

            :param file_contents: (dict) The dictionary where keys are filenames and
                                  values are the contents of the files.
            :return             : (None) The files are indexed with their raw contents.
            '''

            self.indexed_files = file_contents

        def search(self, query):

            '''
            Searches indexed code files for matches using regular expression.

            :param query: (str) The regular expression query to search for
            :return     : (dict) A dictionary where keys are filenames and
                          the values are the matched files to the regex query.
            '''

            matches = {}
            for filename, file_data in self.indexed_files.items():
                if re.search(query, file_data):
                    matches[filename] = file_data
            return matches


class TestingResults:
    pass


#-----Body-----
regex_retriever = SearchTools.RegexRetriever()
data_handler = DataHandler()

for idx, row in data_handler.df.iterrows():
    github_api_url = row["github_api_url"]

    file_contents = data_handler.fetch_code(github_api_url)
    regex_retriever.index_code(file_contents)

    query = re.escape(row["problem_statement"])
    regex_results = regex_retriever.search(query)
    print(f"Regex search results for problem {idx}: {regex_results}")
