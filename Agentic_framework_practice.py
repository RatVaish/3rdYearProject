#-----Imports-----
import pandas as pd
import os
import requests
import tarfile
import io
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np


#----- Classes + Functions -----
class DatabaseHandler:
    def __init__(self, db_path="cleaned_data.csv"):
        self.db_path = db_path
        if os.path.exists(self.db_path):
            self.df = pd.read_csv(self.db_path)
        else:
            self.df = pd.DataFrame(columns=["id", "problem_statement", "github_api_url"])

    def fetch_problems(self):
        '''
        Retrieves the problem_statement from the loaded dataframe.

        :return: (dict)
        '''
        return self.df.to_dict(orient="records")

    def fetch_code(self,github_api_url):

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
                            print(f"Skipping non-text file {member.name}")
                            continue
            return files_content
        else:
            print(f"Failed to retrieve the tarball, Status code: {response.status_code}")
            return {}

    def fetch_answers(self):
        return self.df["patch_filenames"]


class SearchAgent:
    def __init__(self):
        self.search_methods = {}

    def register_search_method(self, name, function):
        self.search_methods[name] = function

    def search(self, method_name, *args, **kwargs):
        if method_name in self.search_methods:
            return self.search_methods[method_name](*args, **kwargs)
        else:
            raise ValueError(f"Search method {method_name} not registered")


class EmbeddingModel:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze().numpy()


class CodeRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.indexed_files = {}
        self.index = None
        self.file_list = {}

    def index_code(self, file_contents):
        self.file_list = list(file_contents.keys())

        embeddings = np.array(
            [self.embedding_model.get_embedding(content).astype(np.float32) for content in file_contents.values()]
        )

        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)

    def search(self, query, top_k=5):
        if self.index is None or len(self.file_list) == 0:
            raise ValueError("Index not initialized")

        query_embedding = self.embedding_model.get_embedding(query).astype(np.float32).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)

        results = [(self.file_list[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return sorted(results, key=lambda x: x[1], reverse=True)


#-----Body-----
db_handler = DatabaseHandler()
embedding_model = EmbeddingModel()
retriever = CodeRetriever(embedding_model)

problems = db_handler.fetch_problems()

if problems and isinstance(problems, list) and len(problems) > 0:
    first_problem = problems[0]
else:
    raise ValueError("No problems found")

file_contents = db_handler.fetch_code(first_problem["github_api_url"])

retriever.index_code(file_contents)

results = retriever.search(first_problem["problem_statement"])
print("Top matching files:", results)
