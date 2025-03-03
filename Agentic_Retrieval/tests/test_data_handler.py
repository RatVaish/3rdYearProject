from Agentic_Retrieval import data_handler

data_control = data_handler.DataHandler("cleaned_data.csv")

print("Fetching Problem Statements:")
problem_statements = data_control.ProblemStatements(data_control.df).fetch()
print(problem_statements.head())

print("\nFetching Github API URLs:")
github_urls = data_control.GithubAPIUrls(data_control.df).fetch()
print(github_urls.head())

example_problem = problem_statements.iloc[0]
print("\nPreprocessed Problem Statement:")
processed_proble
