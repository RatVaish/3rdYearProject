#-----Imports-----
import pandas as pd
import re


#-----Functions-----
def extract_patch_filenames(patch_text):

    '''
    Breaks down the "patch" column from the original dataframe into a list
    of relevant filenames that refer to the location of the problem.

    :param patch_text: (str) Golden Patch from SWE Bench dataset.
    :return          : (list) Filenames from the Golden Patch.
    '''

    if pd.isna(patch_text):
        return []

    pattern = r"diff --git a/([\w./-]+) b/[\w./-]+"
    files = re.findall(pattern, patch_text)
    return list(set(files))


#-----Body-----
df = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet")

df["github_api_url"] = df.apply(lambda row: f"https://api.github.com/repos/{row['repo']}/tarball/{row['base_commit']}", axis=1)
df["patch_files"] = df["patch"].apply(extract_patch_filenames)

df = df.drop(columns=["repo", "instance_id", "base_commit", "patch", "test_patch", "hints_text", "created_at", "version", "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit", "difficulty"])

df_RAG = df.sample(n=50, random_state=42)


print(df_RAG["github_api_url"].head())
#df_RAG.to_csv("cleaned_data_RAG.csv", index=False)
