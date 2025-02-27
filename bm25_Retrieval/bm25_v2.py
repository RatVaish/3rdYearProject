#-----Imports-----
import pandas as pd
import requests
import tarfile
import io
import os
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')


#-----Body-----
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
                        continue
        return files_content
    else:
        print(f"Failed to retrieve the tarball, Status code: {response.status_code}")
        return {}


df = pd.read_csv('../Dev/cleaned_data.csv')

df["extracted_files"] = df["github_api_url"].apply(extract_github_files)

all_files = []
for _, row in df.iterrows():
    problem_statement = row["problem_statement"]
    for file_path, code in row["extracted_files"].items():
        all_files.append({
            "problem_statement": problem_statement,
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "code": code
        })

code_df = pd.DataFrame(all_files)
code_df["tokenized_code"] = code_df["code"].apply(lambda x: tokenizer.tokenize(str(x)))

corpus = code_df["tokenized_code"].tolist()
bm25 = BM25Okapi(corpus)


def retrieve_top_filename(problem_statement):

    '''
    Retrieves the filename of the most relevant python file for a given
    problem statement.

    :param problem_statement: (str) The problem statement
    :return                 : (str) The most relevant filename
    '''

    query_tokens = tokenizer.tokenize(problem_statement)
    scores = bm25.get_scores(query_tokens)
    top_k_indices = sorted(range(len(scores)), key=lambda x: scores[x])
    return code_df.iloc[top_k_indices][["filename"]]


results = []
for _, row in df.iterrows():
    problem_statement = row["problem_statement"]
    best_filename = retrieve_top_filename(problem_statement)

    results.append({
        "problem_statement": problem_statement,
        "matched_filename": best_filename
    })

print(results)