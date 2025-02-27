import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
OPEN_API_KEY = os.getenv('OPEN_API_KEY')
