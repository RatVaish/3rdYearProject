from transformers import AutoTokenizer, AutoModel
import torch
import re
import ast


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
            Indexes the filenames by embedding their contents using the specified
            embedding model.

            :param file_contents: (dict) The dictionary where keys are filenames and
                                  values are the contents of the files.
            :return             : (None) The files are indexed with their raw contents.
            '''

            self.indexed_files = file_contents

        def search(self, query):

            '''
            Searches indexed filenames for matches using regular expression.

            :param query: (str) The regular expression query to search for
            :return     : (str or None) The most relevant filename or None if no matches
            '''

            matches = {filename: len(re.findall(query, filename)) for filename in self.indexed_files}

            # Filter out filenames with no matches
            matches = {filename: count for filename, count in matches.items() if count > 0}

            if matches:
                # Return the filename with the highest match count
                return max(matches, key=matches.get)
            else:
                return None

    class ASTRetriever:
        def __init__(self, file_contents):
            self.file_contents = file_contents

        def search(self, function_name):

            '''
            Searches for a specific function defined in indexed file contents.
            Iterates through file contents and looks for function definitions
            matching the provided function name.

            :param function_name: (str) The function name to search for.
            :return             : (dict) A dictionary where keys are filenames and
                                  values are the contents of the files.
            '''

            results = {}
            for filename, content in self.file_contents.items():
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if node.name == function_name:
                                results[filename] = content
                                break
                except SyntaxError:
                    continue
            return results

    class NewSearch:
        pass
