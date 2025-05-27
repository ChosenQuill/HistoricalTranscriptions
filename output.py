import os

def generate_markdown(directory, output_file):
    """
    Generate a Markdown file containing the contents of all .py files in the specified directory.
    Ignores __pycache__ folders.

    :param directory: The root directory to scan.
    :param output_file: The name of the output Markdown file.
    """
    with open(output_file, "w", encoding="utf-8") as md_file:
        for root, dirs, files in os.walk(directory):
            # Ignore __pycache__ folders
            dirs[:] = [d for d in dirs if d != "__pycache__"]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    md_file.write(f"# {file}\n\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as py_file:
                            content = py_file.read()
                            md_file.write(f"```python\n{content}\n```\n\n")
                    except Exception as e:
                        md_file.write(f"**Error reading file {file}: {e}**\n\n")

if __name__ == "__main__":
    # Replace with your desired directory and output file name
    root_directory = "./src"  # Current directory
    markdown_output = "output.md"
    generate_markdown(root_directory, markdown_output)