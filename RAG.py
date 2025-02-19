#-----Imports-----
import pandas as pd
import spacy
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModel.from_pretrained('microsoft/codebert-base')


#-----Functions-----
def extract_keywords(problem_statement):

    '''
    Extracts keywords (nouns, verbs, etc.) from a problem statement
    using spaCy's NLP capabilities and returns them as a list.

    :param problem_statement: (str) Problem description from SWE Bench
    :return                 : (list) List of keywords extracted from problem statement
    '''

    doc = nlp(problem_statement)
    keywords = set()

    for token in doc:
        if token.pos_ in ["NOUN", "VERB"]:
            keywords.add(token.text.lower())
    return list(keywords)


GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
HEADERS = {
    'Authorization': f"token {GITHUB_TOKEN}",
    "User-Agent": "python-requests"
}


def search_github_repository(keywords, max_results=10):

    '''
    Searches GitHub for relevant code repositories, based on an
    extracted list of keywords from the problem statement.

    :param keywords   : (list) List of keywords
    :param max_results: (int, optional) The maximum number of returned results.
    :return           : (list) List of dictionaries each containing:
                        - "path" (str): The file path of the code file.
                        - "url" (str): The URL of the code file on GitHub.
    '''

    query = "+".join(keywords)
    url = f"https://api.github.com/search/code?q={query}+in:file&per_page={max_results}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"GitHub API request failed with status code {response.status_code}")

    search_results = response.json()

    code_files = []
    for item in search_results["items"]:
        file_path = item["path"]
        file_url = item["url"]
        code_files.append({"path": file_path, "url": file_url})
    return code_files


def get_code_from_github_file(file_url):

    '''
    Retrieves content of code file from GitHub using the associated URL.

    :param file_url: (str) The URL of the code file on GitHub.
    :return        : (str) Content of the code file in string format.
    '''

    response = requests.get(file_url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"Failed to retrieve file content with status code {response.status_code}")

    file_data = response.json()

    content = file_data["content"]
    return content


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


def find_relevant_code_snippets(problem_statement, code_files):

    '''
    Finds the most relevant code snippets using FAISS similarity search to
    compare between problem_statements and retrieved code snippets.

    :param problem_statement: (str) Problem description from SWE Bench
    :param code_files       : (list) List of code files, where each file
                               contains an "url" for GitHub.
    :return                 : (dict) Dictionary containing the "path" and
                               "url" of the code snippet with the highest
                               similarity to the problem statement.
    '''

    problem_embedding = get_embedding(problem_statement)

    code_embeddings = []
    for file in code_files:
        code_context = get_code_from_github_file(file["url"])
        code_embedding = get_embedding(code_context)
        code_embeddings.append(code_embedding)

    problem_embedding = np.vstack(problem_embedding)
    code_embeddings = np.vstack(code_embeddings)

    index = faiss.IndexFlatL2(problem_embedding.shape[1])
    # Can implement other indexes (maybe look into it?)
    index.add(code_embeddings)

    _, indices = index.search(problem_embedding, 1)
    best_match_index = indices[0][0]

    return code_files[best_match_index]


#-----Body-----
df = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet")

problem_statements = df["problem_statement"]

nlp = spacy.load('en_core_web_sm')

results = []

for index, row in df.iterrows():
    problem_statement = row["problem_statement"]

    keywords = extract_keywords(problem_statement)

    code_files = search_github_repository(keywords)

    if not code_files:
        print(f"No relevant code snippets found at index {index}")
        continue

    best_match = find_relevant_code_snippets(problem_statement, code_files)

    results.append({
        "Index": index,
        "Path": best_match["path"],
        "URL": best_match["url"]
    })

    print(f"Most relevant file for issue at index {index}: {best_match}")

df_results = pd.DataFrame(results)

df_results.to_csv("relevant_code_snippets.csv", index=False)
print("Results have been saved to relevant_code_snippets.csv")

df_results.to_excel("relevant_code_snippets.xlsx", index=False)
print("Results have been saved to relevant_code_snippets.xlsx")
