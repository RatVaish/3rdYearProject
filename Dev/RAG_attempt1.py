#-----Imports-----
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import numpy as np


#-----Functions-----

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_codebert_embedding(text):

    '''
    Generates an embedding for the given text using CodeBERT

    This function tokenizes the input text, processes it through CodeBERT,
    and extracts the embedding from the CLS token.

    :param  text: (str)  Input text.
    :return     : (numpy array) 768-dimensional vector representing the embedding.
    '''

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def retrieve_code_snippets(query, top_k=1):

    '''
    Retrieves the most relevant code snippets based on semantic similarity.

    This function converts the query into a CodeBERT embedding, performs a similarity
    search using FAISS, and returns the top-k most relevant code patches.

    :param query: (str) NL query describing issue/fix.
    :param top_k: (int,optional) Number of code snippets to return.
    :return     : (list of dicts) List of dictionaries containing:
                    - "instance_ids": (str) The unique identifier for the retrieved instance.
                    - "patch": (str) Code patch resolving issue.
    '''

    query_embedding = np.array(get_codebert_embedding(query)).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append({
            "instance_id": instance_ids[i],
            "patch": patches[i]
        })
    return results


#-----Body-----
df = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet")

df["combined_text"] = df["problem_statement"] + "\nHints:\n" + df["hints_text"]
df["embedding"] = df["combined_text"].apply(get_codebert_embedding)

first_instance_text = df.iloc[0]["combined_text"]
first_instance_embedding = get_codebert_embedding(first_instance_text)

embeddings = np.array(df["embedding"].tolist()).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

#instance_ids = df["instance_id"].tolist()
#patches = df["patch"].tolist()

instance_ids = [df.iloc[0]["instance_id"]]
patches = [df.iloc[0]["patch"]]

query = "Fix a bug related to database connection failiure"
retrieved_patches = retrieve_code_snippets(query)
print(retrieved_patches)