import json
import hashlib
import os
from tkinter import messagebox
import re

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

def get_project_file(project_path):
    """Get the path to the project.json file."""
    return os.path.join(project_path, "project.json")

def load_project_data(project_path):
    """Load project data including PDFs and segment data."""
    project_file = get_project_file(project_path)
    if os.path.exists(project_file):
        with open(project_file, 'r') as f:
            try:
                project_data = json.load(f)
                # Ensure the data structure is correct
                if "pdfs" not in project_data:
                    project_data["pdfs"] = []
                if "segments" not in project_data:
                    project_data["segments"] = {"pdfs": {}}
                return project_data
            except json.JSONDecodeError:
                messagebox.showerror("Error", f"Failed to parse {project_file}. Starting fresh.")
                return {"pdfs": [], "segments": {"pdfs": {}}}
    else:
        return {"pdfs": [], "segments": {"pdfs": {}}}

def save_project_data(project_path, project_data):
    """Save project data including PDFs and segment data."""
    project_file = get_project_file(project_path)
    with open(project_file, 'w') as f:
        json.dump(project_data, f, indent=2)

def load_project_storage(project_path):
    """Load segment storage data for a specific project (legacy compatibility)."""
    project_data = load_project_data(project_path)
    return project_data.get("segments", {"pdfs": {}})

def save_project_storage(project_path, segment_data):
    """Save segment storage data for a specific project (legacy compatibility)."""
    project_data = load_project_data(project_path)
    project_data["segments"] = segment_data
    save_project_data(project_path, project_data)

# Legacy functions for backward compatibility (deprecated)
def get_project_segments_file(project_path):
    """Legacy function - use get_project_file instead."""
    print("Warning: get_project_segments_file is deprecated. Use get_project_file instead.")
    return get_project_file(project_path)

def load_storage():
    """Legacy function - use load_project_storage instead."""
    # Check if global segments.json exists and warn about migration
    global_segments_file = "segments.json"
    if os.path.exists(global_segments_file):
        print("Warning: Global segments.json detected. Consider migrating to project-specific storage.")
        with open(global_segments_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                messagebox.showerror("Error", f"Failed to parse {global_segments_file}. Starting fresh.")
                return {"pdfs": {}}
    else:
        return {"pdfs": {}}

def save_storage(data):
    """Legacy function - use save_project_storage instead."""
    # Check if global segments.json exists and warn about migration
    global_segments_file = "segments.json"
    print("Warning: Using global segments.json. Consider migrating to project-specific storage.")
    with open(global_segments_file, 'w') as f:
        json.dump(data, f, indent=4)
