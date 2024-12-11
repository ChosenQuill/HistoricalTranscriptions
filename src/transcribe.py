import os
import json
import base64
import requests
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import re

# CONFIGURATION VARIABLES
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o"
EXPORT_FOLDER = Path("export")
TRANSCRIPTS_FOLDER = Path("transcripts")
TRANSCRIPTS_FOLDER.mkdir(exist_ok=True)

TEMPERATURE = 1
TOP_P = 1.0
MAX_COMPLETION_TOKENS = 2000  # Adjust as needed

# HELPER FUNCTIONS

def natural_sort_key(s: str):
    base = os.path.basename(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', base)]

def encode_image_to_base64(image_path: Path) -> str:
    """Reads an image from disk and returns a base64-encoded string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def parse_image_filename(filename: str) -> Tuple[str, int, int]:
    """
    Given a filename like "ASDF_page1_segment1.png", parse out:
    - pdfname = "ASDF"
    - page_number = 1
    - segment_number = 1
    """
    pattern = r"^(.*)_page(\d+)_segment(\d+)\.png$"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    pdfname = match.group(1)
    page_num = int(match.group(2))
    segment_num = int(match.group(3))
    return pdfname, page_num, segment_num

def get_last_three_pages_transcripts(pdfname: str, current_page: int) -> List[str]:
    transcript_file = TRANSCRIPTS_FOLDER / f"{pdfname}.txt"
    if not transcript_file.exists():
        return []

    with open(transcript_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    page_transcripts = {}
    current_page_number = None
    current_page_lines = []

    for line in lines:
        if line.startswith("PAGE "):
            if current_page_number is not None:
                page_transcripts[current_page_number] = "\n".join(current_page_lines)
            try:
                current_page_number = int(line.strip().split(" ")[1])
            except (ValueError, IndexError):
                current_page_number = None
            current_page_lines = []
        else:
            if current_page_number is not None:
                current_page_lines.append(line)

    # Last page accumulation
    if current_page_number is not None:
        page_transcripts[current_page_number] = "\n".join(current_page_lines)

    previous_pages = [p for p in page_transcripts.keys() if p < current_page]
    previous_pages.sort(reverse=True)
    last_3_pages = previous_pages[:3]

    last_3_pages.sort()
    transcripts_to_return = []
    for p in last_3_pages:
        transcripts_to_return.append(f"Previous PAGE {p}:\n{page_transcripts[p]}")

    return transcripts_to_return

def append_page_transcript(pdfname: str, page_number: int, transcript: str) -> None:
    transcript_file = TRANSCRIPTS_FOLDER / f"{pdfname}.txt"
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write(f"PAGE {page_number}\n")
        f.write(transcript.strip() + "\n\n")

def build_prompt_context(pdfname: str, current_page: int, segment_images: List[Path]) -> List[dict]:
    prev_transcripts = get_last_three_pages_transcripts(pdfname, current_page)
    context_block = "\n\n".join(prev_transcripts) if prev_transcripts else "No previous context available."

    image_items = []
    for img_path in segment_images:
        base64_str = encode_image_to_base64(img_path)
        image_items.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_str}"
            }
        })

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant tasked with transcribing historical documents. "
            "You must accurately extract all textual content from the provided image segments. "
            "These images are scans of historical documents and may contain faded ink, unusual fonts, or damage. "
            "Use the provided previous pages' transcripts as context if it helps you interpret unclear text. "
            "However, DO NOT HALLUCINATE. If something is unreadable, mark it as [unreadable]. "
            "Preserve line breaks if meaningful. "
            "DO NOT ADD EXTRANEOUS COMMENTARY, ONLY OUTPUT THE RAW TRANSCRIPTION TEXT. "
            "Do not add page headers in your final output. Your goal: produce the most accurate transcription."
        )
    }

    user_msg_content = [
        {
            "type": "text",
            "text": (
                "Below are historical document segments. Transcribe them as accurately as possible. "
                "DO NOT ADD EXTRA OUTPUT, ONLY OUTPUT THE RAW TRANSCRIPTION TEXT ALONE."
                "Use the previous pages' context to help interpret unclear words if possible.\n\n"
                f"Previous context for {pdfname}, up to the last 3 pages before page {current_page}:\n"
                f"{context_block}\n\n"
                "Now here are the images to transcribe:"
            )
        }
    ]
    user_msg_content.extend(image_items)

    user_msg = {
        "role": "user",
        "content": user_msg_content
    }

    return [system_msg, user_msg]

def call_api(messages: List[dict]) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_completion_tokens": MAX_COMPLETION_TOKENS
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")

    resp_json = response.json()
    choices = resp_json.get("choices", [])
    if not choices:
        raise RuntimeError("No choices returned from API.")
    return choices[0]["message"]["content"].strip()

# MAIN SCRIPT

def main():
    pdf_pages = defaultdict(lambda: defaultdict(list))

    # Collect all segment images
    for img_file in EXPORT_FOLDER.glob("*.png"):
        pdfname, page_num, segment_num = parse_image_filename(img_file.name)
        pdf_pages[pdfname][page_num].append(img_file)

    # Sort the PDF names using natural sort
    pdf_names = sorted(pdf_pages.keys(), key=natural_sort_key)

    for pdfname in pdf_names:
        # Check if transcript for this pdf already exists
        transcript_file = TRANSCRIPTS_FOLDER / f"{pdfname}.txt"
        if transcript_file.exists():
            # Skip this PDF entirely as it was already transcribed in a previous run
            print(f"Skipping {pdfname} as transcript already exists.")
            continue

        pages = pdf_pages[pdfname]
        # Sort pages naturally as well (though they are numeric, we'll just use sorted keys)
        for page_num in sorted(pages.keys()):
            segment_images = pages[page_num]
            # Ensure segments are sorted by segment number
            segment_images = sorted(segment_images, key=lambda p: int(re.search(r"_segment(\d+)\.png$", p.name).group(1)))

            messages = build_prompt_context(pdfname, page_num, segment_images)

            print(f"Processing {pdfname} page {page_num} with {len(segment_images)} segments...")
            transcription = call_api(messages)

            append_page_transcript(pdfname, page_num, transcription)
            print(f"Transcript for {pdfname} page {page_num} saved.")

    print("All PDF pages processed. Transcriptions complete.")

if __name__ == "__main__":
    main()
