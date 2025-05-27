import json
import hashlib
import os
from tkinter import messagebox
from utils.config import STORAGE_FILE
import re
import os

# Used for sorting filenames based on natural sort key.
def natural_sort_key(s):
    base = os.path.basename(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', base)]

def compute_pdf_hash(pdf_path):
    hash_sha256 = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def load_storage():
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                messagebox.showerror("Error", f"Failed to parse {STORAGE_FILE}. Starting fresh.")
                return {"pdfs": {}}
    else:
        return {"pdfs": {}}

def save_storage(data):
    with open(STORAGE_FILE, 'w') as f:
        json.dump(data, f, indent=4)
