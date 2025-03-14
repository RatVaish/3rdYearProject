#----Imports-----
import pandas as pd
import requests
import tarfile
import io
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

# Load environment variables from keys.env
load_dotenv("keys.env")

tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')


#-----Body-----
def extract_github_filenames(github_api_url):

    """
    Extracts only filenames from a GitHub repository tarball URL.

    :param github_api_url: (str) The  GitHub API url for the tarball of a
                            repository at a specific commit hash.
    :return              : (list) A list of extracted filenames.
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
                if member.isfile() and member.name.endswith(".py"):
                    filenames.append(os.path.basename(member.name))
        return filenames
    else:
        print(f"Failed to retrieve the tarball, Status code: {response.status_code}")
        return []


df = pd.read_csv('cleaned_data.csv')

df["retrieved_filenames"] = df["github_api_url"].apply(extract_github_filenames)

all_files = []
for _, row in df.iterrows():
    problem_statement = row["problem_statement"]
    for filename in row["retrieved_filenames"]:
        all_files.append({
            "problem_statement": problem_statement,
            "filename": filename
        })

filename_df = pd.DataFrame(all_files)

filename_df["tokenized_filename"] = filename_df["filename"].apply(lambda x: tokenizer.tokenize(str(x)))

corpus = filename_df["tokenized_filename"].tolist()
bm25 = BM25Okapi(corpus)


def retrieve_top_filename(problem_statement):

    """
    Retrieves the filename of the most relevant python file for a given
    problem statement.

    :param problem_statement: (str) The problem statement
    :return                 : (str) The most relevant filename
    """

    query_tokens = tokenizer.tokenize(problem_statement)
    scores = bm25.get_scores(query_tokens)
    top_index = scores.argmax()
    return filename_df.iloc[top_index][["filename"]]


results = []
for _, row in df.iterrows():
    problem_statement = row["problem_statement"]
    best_filename = retrieve_top_filename(problem_statement)

    results.append({
        "problem_statement": problem_statement,
        "matched_filename": best_filename,
        "patch_filename": row["patch_files"]
    })

    print(f"matched_filename: {best_filename}")
    print(f"patch_filename: {row['patch_files']}")
    print("-" * 50)

results_df_full = pd.DataFrame(results)
results_df_full.to_csv("results_full_bm25.csv")
