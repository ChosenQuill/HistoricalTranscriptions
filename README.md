# Historical Transcriptions

This project provides a user-assisted pipeline for segmenting and transcribing historical document scans. It is particularly tailored to assist historians, archivists, and researchers who need to process large collections of scanned historical records, often with irregular formatting, damage, or discoloration, into machine-readable text.

Originally developed in collaboration with Dr. Wright, who was researching 19th-century missionaries from Sierra Leone and Liberia, this tool helped to produce over 150+ pages of accurately transcribed historical documents. By integrating both computer vision techniques and AI-based transcription (including state-of-the-art large language models), the project aims to streamline and improve the accuracy of manual transcription efforts.

## Overview

Historical documents, especially century-old records, often come in irregular shapes, skewed scans, or damaged pages. Relying on fully automated segmentation and transcription methods frequently fails to yield accurate results due to inconsistent page layouts and scanning artifacts.

This toolkit aims to combine human-assisted segmentation with automated transcription:

1. **Human-Assisted Segmentation:**
    
    The user can visually review each scanned PDF page, define "scan pages" (logical subdivisions of a single PDF page image), and mark segments within each scan page using a GUI tool. These user-defined segments help isolate blocks of text or distinct records from cluttered or irregular page scans.
    
2. **Automated Transcription:**
    
    Once the segments are defined and exported, the transcription tool applies OCR through multi modal large language models to convert the segmented images into text. The final result is a cleaner, more reliable transcription of the historical document.
    

This approach ensures high accuracy and preserves contextual integrity in text extraction, making it easier for researchers to analyze large sets of historical data.

## Workflow Overview

### Phase 1: Segmenting the Pages (using `main.py`)

1. **Load PDFs**:
    
    Place your scanned PDF files into the `docs` directory. The application will detect and list them automatically.
    
2. **Start the Application**:
    
    Run `src/main.py` after setting up the environment (detailed instructions below). A GUI will appear, displaying each PDF page sequentially.
    
3. **Defining Segments**:
    - Navigate through the PDF pages using the provided buttons (e.g., *Next Page*, *Prev Page*).
    - Click-and-drag (or click in four corners) to define rectangular segments of the document you want to extract. You can switch between "add" mode (defining new segments) and "edit" mode (adjusting corners) to perfectly capture even skewed or irregular text blocks.
    - Press hotkeys to rapidly remove the last segment, clear all segments, or split a segment into multiple sub-segments if needed.
4. **Exporting Segments**:
    
    Once you're satisfied with the segmentation of a page, hit the *Export Segments & Next* button to save the extracted segments as images into the `export` folder. 
    

### Phase 2: Transcribing the Segments (using `transcribe.py`)

1. **AI Transcription**:
    
    Run `src/transcribe.py` after your segments have been exported. This script reads each exported segment image and uses the OpenAI API (GPT-4o) to produce a transcription.
    
2. **Storing Results**:
    
    The transcriptions, along with their corresponding metadata (page numbers, segment identifiers, etc.), are saved in the `final` directory, creating a ready-to-use corpus for further historical or textual analysis.
    

### Experimental Scripts

- **`segment.py`**: Uses computer vision to try automated segmentation. Good as a starting point or to assist in batch-processing. Results are experimental and may still require manual refinement.
- **`segment_tess.py`**: Integrates Tesseract OCR for a different automated approach to segmentation. While often less accurate than the user-assisted method, it can still provide a starting baseline.

## Installation and Setup

### Requirements

- **Python 3.9+** (recommended)
- **Poetry** for dependency management
- An **OpenAI API Key** for the transcription step

### Steps to Install

1. **Clone the Repository**:
    
    ```bash
    git clone https://github.com/username/historical-doc-transcription.git
    cd historical-doc-transcription
    ```
    
2. **Install Dependencies**:
    
    ```bash
    poetry install
    sudo apt-get install python3-tk
    ```
    
3. **Set Up Environment Variables**:
    
    Create a `.env` file in the project root and add your OpenAI API key:
    
    ```
    OPENAI_API_KEY=sk-yourapikey...
    ```
    
4. **Prepare the Directories**:
    
    Ensure that `docs`, `export`, and `final` directories exist. `docs` will contain input PDFs, `export` will hold segmented image files, and `final` will contain the final transcriptions and output data.
    

## Usage

### Running the GUI Segmenter

```bash
poetry run python3 src/main.py
```

- Use the on-screen buttons to navigate through PDFs.
- Drag to select new segments or switch to edit mode to refine segment boundaries.
- Press hotkeys (e.g., `A` and `S` for navigation, `Shift` to remove last segment, `C` to clear, `R` to export and move to the next page, etc.) to speed up the workflow.
- Once you have segmented all pages, all cutout images will be available in `export`.

### Running the Transcription Script

```bash
poetry run python3 src/transcribe.py
```

- This reads each exported segment from `export`, calling the OpenAI API to generate a transcript for each segment.
- Transcripts are saved in `final`.

### Running Experimental Scripts

- **Automated Segmentation with Computer Vision**:
    
    ```bash
    poetry run python3 segment.py
    ```
    
- **Automated Segmentation with Tesseract**:
    
    ```bash
    poetry run python segment_tess.py
    ```
    

Note these scripts are experimental and require further refinement for optimal output.