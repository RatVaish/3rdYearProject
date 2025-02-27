#-----Imports------
import pandas as pd
import requests
import tarfile
import io
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize model and tokenizer for CodeBERT
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModel.from_pretrained('microsoft/codebert-base')


#-----Functions-----
def extract_github_files(github_api_url):

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
                        print(f"Skipping non-text file {member.name}")
                        continue
        return files_content
    else:
        print(f"Failed to retrieve the tarball, Status code: {response.status_code}")
        return {}


def get_embedding(text):

    '''
    Uses the CodeBERT model to generate embeddings got a given text. This
    applies to both the problem statement and code file.

    :param text: (str) The input text that an embedding is generated for.
    :return    : (numpy.ndarray) Vector representation of the input text.
    '''

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :].numpy()
    return embeddings


def similarity_search(problem_statement, github_files):

    '''
    Performs a similarity search between the problem statement and the content of the
    project files using FAISS, returning the most relevant file.

    :param problem_statement: (str) The problem statement from the SWE Bench Dataset
    :param github_files     : (dict) A dictionary where keys are file paths and
                                values are the contents of the project files.
    :return                 : (str) The name of the file that is most similar to the
                                problem statement.
    '''

    problem_embedding = get_embedding(problem_statement)

    file_embeddings = []
    file_names = []

    for file_name, file_content in github_files.items():
        file_embedding = get_embedding(file_content)
        file_embeddings.append(file_embedding)
        file_names.append(file_name)

    problem_embedding = np.vstack(problem_embedding)
    file_embeddings = np.vstack(file_embeddings)

    index = faiss.IndexFlatL2(problem_embedding.shape[1])
    index.add(file_embeddings)

    _, indices = index.search(problem_embedding, 1)
    best_match_index = indices[0][0]

    return file_names[best_match_index]


#------Body------
df = pd.read_csv('../Dev/cleaned_data.csv')

# This code runs RAG for the first element of the DataFrame
'''
row = df.iloc[0]

github_api_url = row['github_api_url']
problem_statement = row['problem_statement']

github_files = extract_github_files(github_api_url)

best_match_file = similarity_search(problem_statement, github_files)

print(f"Most relevant file: {best_match_file}")
print("-" * 50)
'''

# This code is for running the RAG for all 500 elements of the Dataframe
for index,row in df.iterrows():
    github_api_url = row['github_api_url']
    problem_statement = row['problem_statement']

    github_files = extract_github_files(github_api_url)

    best_match_file = similarity_search(problem_statement, github_files)

    print(f"Most relevant file: {best_match_file}")
    print(f"Patch file: {row['patch_file']}")
    print("-" * 50)
