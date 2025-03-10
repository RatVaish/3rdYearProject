import os
from dotenv import load_dotenv

# Load environment variables from keys.env
load_dotenv("keys.env")

# Fetch the token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Debugging: Print a masked version of the token to check if it's loaded
if GITHUB_TOKEN:
    print(f"GITHUB_TOKEN loaded: {GITHUB_TOKEN[:5]}*********")
else:
    print("GITHUB_TOKEN is NOT loaded. Check your keys.env file.")
