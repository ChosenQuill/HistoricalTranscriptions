import os
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"
EXPORT_DIR = "export"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
