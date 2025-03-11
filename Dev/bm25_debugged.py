#----Imports-----
import pandas as pd
import requests
import tarfile
import io
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

# Load environment variables
load_dotenv("keys.env")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')


#-----Body-----
def extract_github_filenames(github_api_url):
    """
    Extracts only Python filenames (.py) from a GitHub repository tarball URL.
    """
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

        print(f"Extracted {len(filenames)} Python files from {github_api_url}")
        return filenames

    else:
        print(f"Failed to retrieve tarball from {github_api_url}, Status code: {response.status_code}")
        return []


# Load dataset
df = pd.read_csv('cleaned_data_RAG.csv')

# Extract Python filenames
df["retrieved_filenames"] = df["github_api_url"].apply(extract_github_filenames)

# Convert extracted filenames into a structured DataFrame
all_files = []
for _, row in df.iterrows():
    problem_statement = row["problem_statement"]
    for filename in row["retrieved_filenames"]:
        all_files.append({
            "problem_statement": problem_statement,
            "filename": filename
        })

filename_df = pd.DataFrame(all_files)

if filename_df.empty:
    print("No filenames were extracted. BM25 cannot be initialized.")
    exit()

# Tokenize filenames
filename_df["tokenized_filename"] = filename_df["filename"].apply(lambda x: tokenizer.tokenize(str(x)))

# Debug: Print tokenized filenames
print("Sample filenames and their tokens:")
print(filename_df.head(5)[["filename", "tokenized_filename"]])

# Initialize BM25 (ensuring correct input format)
corpus = filename_df["tokenized_filename"].tolist()
corpus = [tokens if isinstance(tokens, list) else [] for tokens in corpus]

if not corpus:
    print("BM25 corpus is empty. Exiting...")
    exit()

bm25 = BM25Okapi(corpus)


def retrieve_top_filename(problem_statement):
    """
    Retrieves the filename of the most relevant Python file for a given problem statement.
    """
    query_tokens = tokenizer.tokenize(problem_statement)
    scores = bm25.get_scores(query_tokens)

    if len(scores) == 0:
        return "No relevant file found"

    top_index = scores.argmax()
    return filename_df.iloc[top_index]["filename"]  # ✅ Return a string, not a Series


# Apply BM25 ranking to find the best filename
results = []
for _, row in df.iterrows():
    problem_statement = row["problem_statement"]
    best_filename = retrieve_top_filename(problem_statement)

    results.append({
        "problem_statement": problem_statement,
        "matched_filename": best_filename
    })

    print(f"Matched Filename: {best_filename}")
    print("-" * 50)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("results_df_RAG.csv", index=False)
