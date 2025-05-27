import os
import re
import cv2
import numpy as np
import math
from glob import glob

"""
This script is an experimental utility tool takes scanned PDF pages (as images) from the export folder 
and automatically divides each page into equally sized horizontal segments. However, rather than blindly 
cutting at exact intervals, it will attempt to avoid cutting through lines of text by using computer vision 
techniques to find gap areas between lines of text.

It will:
1. Load all images named "{pdf_name}_page{page_number}.png" from the export folder.
2. For each page, determine how many segments to produce or the desired segment height.
   (For simplicity, define `NUM_SEGMENTS` or `SEGMENT_HEIGHT` below.)
3. Convert the image to grayscale, binarize it, and use morphological operations to highlight
   text regions.
4. Compute a horizontal projection profile to identify areas with text (high pixel density) and 
   areas without text (low pixel density).
5. Starting with equal-height cut lines, try to nudge each cut line slightly up or down to find a 
   low-text-density area to make a cleaner cut.
6. Save each segment as a new image file, named "{pdfname}_page{pagenum}_segment{i}.png".

Note: This approach does not rely on OCR since historical documents are of poor quality. Instead,
it uses image processing and projection profiles to find safer cut lines.

Adjustable parameters:
- NUM_SEGMENTS: number of horizontal segments per page (if defined).
  OR
- SEGMENT_HEIGHT: fixed height of each segment (if defined).
- MAX_LINE_ADJUST: how many pixels up or down we can nudge a cut line to avoid text.
- THRESH_BIN: binarization threshold method (Otsu by default).
- KERNEL_SIZE: morphological kernel size.
- MIN_GAP_WEIGHT: how to define a good gap line (pick line with minimal text pixels in a window).
"""

EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# USER-DEFINABLE PARAMETERS
# Either define a fixed number of segments or a segment height in pixels.
NUM_SEGMENTS = 5  # Divide each page into 5 equal-height segments
SEGMENT_HEIGHT = None  # If specified, overrides NUM_SEGMENTS. e.g. SEGMENT_HEIGHT = 1000

# How far we can adjust the boundary line to find a better gap
MAX_LINE_ADJUST = 30

# Morphological kernel size for text enhancement
KERNEL_SIZE = (3,3)

def natural_sort_key(s):
    base = os.path.basename(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', base)]

def find_best_cut_line(projection, initial_line, max_adjust):
    """
    Given a horizontal projection (sum of black pixels per row), and an initial cut line,
    try shifting the line up or down within 'max_adjust' range to find the position 
    with minimal text presence.
    """
    best_line = initial_line
    best_val = projection[initial_line] if 0 <= initial_line < len(projection) else float('inf')

    # Search range is from initial_line-max_adjust to initial_line+max_adjust
    start_line = max(0, initial_line - max_adjust)
    end_line = min(len(projection)-1, initial_line + max_adjust)

    for line_pos in range(start_line, end_line+1):
        val = projection[line_pos]
        if val < best_val:
            best_val = val
            best_line = line_pos

    return best_line

def process_page(image_path):
    # Extract base name: {pdfname}_page{pagenum}.png
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    match = re.match(r"(.*)_page(\d+)$", base)
    if not match:
        # If does not match the pattern, try a simpler pattern
        match = re.match(r"(.*)_page(\d+)", base)
    if not match:
        # If still no match, just treat the whole name as pdf name and no page number
        pdf_name = base
        page_num = 1
    else:
        pdf_name, page_num = match.group(1), match.group(2)
    page_num = int(page_num)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize image using Otsu's threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so text = white, background = black (for easier morphological ops)
    # This way we can sum white pixels in projection to find text lines.
    inverted = cv2.bitwise_not(thresh)

    # Morphological closing to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    h, w = morph.shape

    # Compute horizontal projection: sum of white pixels in each row
    # White pixels represent text regions, so a row with low sum = good gap
    projection = np.sum(morph == 255, axis=1)

    # Determine the number of segments and the height per segment
    if SEGMENT_HEIGHT is not None:
        segment_height = SEGMENT_HEIGHT
        num_segments = math.ceil(h / segment_height)
    else:
        num_segments = NUM_SEGMENTS
        segment_height = h / num_segments

    # Initial cut lines (not including the top = 0 and bottom = h)
    # We'll cut after each segment, so lines are at multiples of segment_height
    # Example: for 5 segments, lines at segment_height, 2*segment_height, ...
    cut_lines = []
    for i in range(1, num_segments):
        line_pos = int(round(i * segment_height))
        # Find a better line position near this line_pos
        best_line = find_best_cut_line(projection, line_pos, MAX_LINE_ADJUST)
        cut_lines.append(best_line)

    # Add the boundaries of the page
    segment_boundaries = [0] + cut_lines + [h]

    # Export each segment
    for i in range(len(segment_boundaries)-1):
        top = segment_boundaries[i]
        bottom = segment_boundaries[i+1]
        segment_img = img[top:bottom, :]
        out_name = f"{pdf_name}_page{page_num}_segment{i+1}.png"
        out_path = os.path.join(EXPORT_DIR, out_name)
        cv2.imwrite(out_path, segment_img)
        print(f"Exported: {out_path}")

def main():
    image_paths = sorted(glob(os.path.join(EXPORT_DIR, "*_page*.png")), key=natural_sort_key)
    if not image_paths:
        print("No page images found in the export folder.")
        return

    for path in image_paths:
        print(f"Processing {path}...")
        process_page(path)

if __name__ == "__main__":
    main()
