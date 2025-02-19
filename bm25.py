# -----Imports-----
import pandas as pd
from rank_bm25 import BM25Okapi
import json


# -----Functions-----
def process_json_field(field):
    '''
    Converts json fields into str fields
    :param  field(json):
    :return field(str):
    '''
    if isinstance(field, str):
        try:
            return " ".join(json.loads(field))
        except json.JSONDecodeError:
            return field
        return ""


def clean_text(text):
    '''
    Handles missing or null values
    :param  input_text:
    :return cleaned_text:
    '''
    if not text or text.strip() == "":
        return "unknown"
    return text


def construct_document(column):

    '''

    Creates a BM-25 Compatible document for retrieval by constructing a text representation for each instance
    :param  column:
    :return       :

    '''

    problem = clean_text(column["problem_statement"])
    hints = clean_text(column["hints_text"])
    patch = clean_text(column["patch"])
    test_patch = clean_text(column["test_patch"])
    fail_to_pass = clean_text(column["FAIL_TO_PASS"])
    pass_to_pass = clean_text(column["PASS_TO_PASS"])

    document = f"{problem} {hints} {patch} {test_patch} {fail_to_pass} {pass_to_pass}"
    return document


def retrieve_code_snippets(query, bm25, df, top_n=5):
    '''
    Retrieves relevant code Snippets for any given problem statement
    :param  query:
    :param  bm25:
    :param  df:
    :param  top_n:
    :return filtered_df:
    '''
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = scores.argsort()[-top_n:][::-1]

    return df.iloc[top_indices][["instance_id", "patch", "bm25_doc"]]


# -----Body-----
df = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet")

df["bm25_doc"] = df.apply(construct_document, axis=1)
df["bm25_doc_tokens"] = df["bm25_doc"].apply(lambda x: x.lower().split())

bm25 = BM25Okapi(df["bm25_doc_tokens"].tolist())

query = "Fix bug in database connection handling"
results = retrieve_code_snippets(query, bm25, df)
print(results)
results.shape