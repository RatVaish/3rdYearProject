#-----Imports-----
import os
from agent import CodeRetrievalAgent
import pandas as pd


#-----Main Function-----
def main():
    db_path = "cleaned_data.csv"
    agent = CodeRetrievalAgent(db_path=db_path, use_llm=True)

    answers = []
    for index in range(len(agent.problem_statement.fetch())):
        print(f"Processing issue {index + 1}/{len(agent.problem_statement.fetch())}...")

        # Run the agent for the current problem statement
        best_filename = agent.run(index)

        # Display results
        if best_filename:
            print(f"✅ Best match for issue {index + 1}: {best_filename}")
            answers.append(best_filename)
        else:
            print(f"❌ No match found for issue {index + 1}")
            answers.append(None)

    df_answers = pd.DataFrame(answers)
    df_answers.to_csv("retrieved_filenames_agent.csv", index=False)
    print("Results have been saved to retrieved_filenames_agent.csv")


if __name__ == "__main__":
    main()
