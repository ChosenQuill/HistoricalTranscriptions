# auto_segment.py

```python
import os
import re
import cv2
import numpy as np
import math
from glob import glob
import fitz  # PyMuPDF for PDF reading
from pdf2image import convert_from_path

DOCS_DIR = "docs"
EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Parameters
LINES_PER_SEGMENT = 5  # Target ~5 lines per segment
MAX_ANGLE = 45         # Ignore lines with angles > 45 degrees
MORPH_SIZE_FOR_LINES = (25, 1)  # Horizontal morphological kernel to link text into lines
MIN_PAGE_AREA = 50000   # Minimum area to consider a detected block as a page

def natural_sort_key(s):
    base = os.path.basename(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', base)]

def deskew_image(img):
    """
    Deskew the image by detecting dominant text lines angle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold for possibly more stable binarization on historical docs
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 25, 15)
    bw = cv2.bitwise_not(bw)

    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            for rho, theta in line:
                deg = (theta * 180 / np.pi)
                if deg > 90:
                    deg -= 180
                if abs(deg) < MAX_ANGLE:
                    angles.append(deg)
        if angles:
            angle = np.median(angles)

    if abs(angle) > 0.1:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), 
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img

def detect_pages(img):
    """
    Detect "scanned pages" within a PDF page image using connected components analysis.
    Steps:
    1. Convert to grayscale, threshold.
    2. Morphological closing to merge text into large blocks.
    3. Find connected components.
    4. Filter large blocks as pages.
    5. If a block is not roughly rectangular, approximate polygon and try perspective correction.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use adaptive threshold for possibly uneven lighting
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    inv = cv2.bitwise_not(bw)

    # Morphological close to get bigger blocks (merge text into paragraphs)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)

    # Connected components
    # We want to find large blocks that could represent a page
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    pages = []
    h, w = closed.shape
    for i in range(1, num_labels):
        x, y, w_, h_ = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area > MIN_PAGE_AREA and w_ > 100 and h_ > 100:
            candidate = img[y:y+h_, x:x+w_]
            # Attempt a slight perspective correction if needed
            # We could try edge detection to find a quadrilateral
            candidate = perspective_correction(candidate)
            candidate = deskew_image(candidate)
            if candidate is not None and candidate.size > 0:
                pages.append(candidate)

    # If no pages found, fallback to using the entire image as one page
    if not pages:
        full = deskew_image(img)
        if full is not None and full.size > 0:
            pages = [full]

    return pages

def perspective_correction(img):
    """
    Attempt a rough perspective correction by detecting edges and approximating a polygon.
    If fail, return the image as is.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_approx = None
    best_area = 0
    h, w = gray.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.5 * w * h:  # We want a contour that covers a large fraction of the image
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_approx = approx

    if best_approx is not None:
        pts = best_approx.reshape(4,2).astype(np.float32)
        pts = order_points(pts)
        (tl, tr, br, bl) = pts

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth-1, 0],
            [maxWidth-1, maxHeight-1],
            [0, maxHeight-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped
    return img

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_text_lines(img):
    """
    Find horizontal text lines by linking characters:
    1. Binarize & invert: text = white
    2. Morphological dilation horizontally to connect characters into lines.
    3. Find the y-coordinates of each line by projection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarize
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(th)

    # Dilate horizontally to connect text into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_SIZE_FOR_LINES)
    dil = cv2.dilate(inv, kernel, iterations=1)

    # Projection
    h, w = dil.shape
    projection = np.sum(dil==255, axis=1)
    # A line of text should have a significant number of white pixels
    mean_val = np.mean(projection)
    line_indices = []
    # Identify rows that have more than mean_val/2 pixels as potential line centers
    # Then group consecutive rows together as a single line region.
    is_line = projection > (mean_val/2)
    start = None
    for i, val in enumerate(is_line):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            line_indices.append((start, end))
            start = None
    if start is not None:
        line_indices.append((start, h-1))

    # Get line centers as the midpoint of these indices
    line_positions = []
    for (st, en) in line_indices:
        line_positions.append((st+en)//2)

    return line_positions

def segment_lines_into_groups(img, line_positions, pdf_name, scan_page_number):
    """
    Group lines into segments. Each segment will contain approximately LINES_PER_SEGMENT lines.
    We'll create segments by cutting between groups of lines rather than at arbitrary heights.
    """
    # If no lines, just save the whole image as one segment
    if not line_positions:
        out_name = f"{pdf_name}_page{scan_page_number}_segment1.png"
        out_path = os.path.join(EXPORT_DIR, out_name)
        if img.size > 0:
            cv2.imwrite(out_path, img)
            print(f"Exported: {out_path}")
        return

    # Sort line positions
    line_positions = sorted(line_positions)
    h, w = img.shape[:2]

    # We'll form segments by taking approximately LINES_PER_SEGMENT lines at a time
    segments = []
    i = 0
    while i < len(line_positions):
        # Start line
        start_line = line_positions[i]
        # End line (after LINES_PER_SEGMENT lines or the last line)
        end_idx = min(i+LINES_PER_SEGMENT-1, len(line_positions)-1)
        end_line = line_positions[end_idx]

        # Add a small margin above the start_line and below the end_line
        top = max(start_line - 20, 0)
        bottom = min(end_line + 20, h)
        segments.append((top, bottom))
        i = end_idx + 1

    # If for some reason we got no segments, fallback to entire image
    if not segments:
        segments = [(0,h)]

    # Export segments
    for seg_i, (top, bottom) in enumerate(segments, start=1):
        if bottom > top:
            segment_img = img[top:bottom, :]
            if segment_img.size > 0:
                out_name = f"{pdf_name}_page{scan_page_number}_segment{seg_i}.png"
                out_path = os.path.join(EXPORT_DIR, out_name)
                cv2.imwrite(out_path, segment_img)
                print(f"Exported: {out_path}")
        else:
            print(f"Warning: invalid segment boundaries ({top}, {bottom}) for {pdf_name}_page{scan_page_number}_segment{seg_i}")

def process_pdf(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = convert_from_path(pdf_path, dpi=300)

    for page_index, page_image in enumerate(pages, start=1):
        page_cv = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        scanned_pages = detect_pages(page_cv)
        if not scanned_pages:
            # fallback to the entire page if no sub-pages detected
            scanned_pages = [deskew_image(page_cv)]

        for sp_i, sp in enumerate(scanned_pages, start=1):
            if sp is not None and sp.size > 0:
                line_positions = find_text_lines(sp)
                segment_lines_into_groups(sp, line_positions, pdf_name + f"_page{page_index}", sp_i)
            else:
                print(f"Scanned page {sp_i} in {pdf_name}_page{page_index} is empty or invalid. Skipping.")

def main():
    pdf_files = glob(os.path.join(DOCS_DIR, "*.pdf"))
    pdf_files.sort(key=natural_sort_key)

    if not pdf_files:
        print("No PDFs found in 'docs' folder.")
        return

    for pdf_file in pdf_files:
        print(f"Processing PDF: {pdf_file}")
        process_pdf(pdf_file)

if __name__ == "__main__":
    main()

```

# segment_tess.py

```python
# Experimental proof of concept script where we try using pytesseract to create segments, not used in final application as fairly buggy. Uses terreract v5. 

import os
import glob
import math
import hashlib
import fitz
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output

DOCS_DIR = "docs"
EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)

DPI = 300
LINES_PER_SEGMENT = 5

# Angle sweep parameters
ANGLE_RANGE = 5   # Test angles from -5 to 5 degrees
ANGLE_STEP = 1    # Step in degrees

def pdf_to_images(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    imgs = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
    return imgs

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess(img):
    """
    Preprocessing to improve OCR accuracy:
    - Convert to grayscale
    - Otsu binarization
    - Morphological operations to reduce noise
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Optional morphological operations
    # Close to connect text parts (adjust kernel size depending on text density)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert back to BGR for Tesseract
    processed = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    return processed

def extract_lines_with_hierarchy(img):
    """
    Use Tesseract and consider block_num, par_num, line_num to form accurate lines.
    """
    custom_config = r'--psm 4 --oem 1'
    data = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config)

    line_groups = {}
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if text == "" or text.isspace():
            continue
        block = data['block_num'][i]
        par = data['par_num'][i]
        line = data['line_num'][i]
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        key = (block, par, line)
        if key not in line_groups:
            line_groups[key] = {
                "x1": x, "y1": y,
                "x2": x+w, "y2": y+h
            }
        else:
            G = line_groups[key]
            G["x1"] = min(G["x1"], x)
            G["y1"] = min(G["y1"], y)
            G["x2"] = max(G["x2"], x+w)
            G["y2"] = max(G["y2"], y+h)

    line_boxes = []
    for _, v in line_groups.items():
        line_boxes.append((v["x1"], v["y1"], v["x2"], v["y2"]))

    # Sort lines by top coordinate
    line_boxes.sort(key=lambda b: (b[1], b[0]))

    # Optionally filter out extremely large boxes (heuristic)
    # For example, if a line is extraordinarily wide or tall
    filtered = []
    if line_boxes:
        median_width = np.median([b[2]-b[0] for b in line_boxes])
        median_height = np.median([b[3]-b[1] for b in line_boxes])
        for b in line_boxes:
            w = b[2]-b[0]
            h = b[3]-b[1]
            # Simple heuristic: discard lines that are more than 3x median width or height
            if w <= 3*median_width and h <= 3*median_height:
                filtered.append(b)
        line_boxes = filtered if filtered else line_boxes

    return line_boxes

def measure_line_alignment(line_boxes):
    """
    Measure how horizontally aligned lines are by checking the variance of their vertical midpoints.
    Lower variance = more aligned horizontally.
    """
    if not line_boxes:
        return float('inf')
    # Compute vertical midpoints of each line
    midpoints = []
    for (x1, y1, x2, y2) in line_boxes:
        mid = (y1+y2)/2.0
        midpoints.append(mid)
    if len(midpoints) < 2:
        return float('inf')
    return np.var(midpoints)

def find_best_angle(img):
    """
    Try angles from -ANGLE_RANGE to ANGLE_RANGE to find the best deskew angle.
    We use measure_line_alignment to pick the angle with the lowest variance.
    """
    best_angle = 0
    best_variance = float('inf')

    # Preprocess once, then rotate that preprocessed image
    # Actually we must preprocess after rotation since rotation changes orientation
    # But to speed up, we can rotate first then preprocess inside the loop
    for angle in range(-ANGLE_RANGE, ANGLE_RANGE+1, ANGLE_STEP):
        rotated = rotate_image(img, angle)
        processed = preprocess(rotated)
        line_boxes = extract_lines_with_hierarchy(processed)
        var = measure_line_alignment(line_boxes)
        if var < best_variance:
            best_variance = var
            best_angle = angle

    return best_angle

def group_lines_into_segments(line_boxes, lines_per_segment=5):
    segments = []
    for i in range(0, len(line_boxes), lines_per_segment):
        chunk = line_boxes[i:i+lines_per_segment]
        x1 = min(b[0] for b in chunk)
        y1 = min(b[1] for b in chunk)
        x2 = max(b[2] for b in chunk)
        y2 = max(b[3] for b in chunk)

        segments.append({
            "segment_id": i//lines_per_segment + 1,
            "bbox": (x1, y1, x2, y2)
        })
    return segments

def export_segment_images(pdf_name, page_number, final_img, segments):
    base_name = os.path.splitext(pdf_name)[0]
    h, w = final_img.shape[:2]

    for seg in segments:
        seg_id = seg["segment_id"]
        x1, y1, x2, y2 = seg["bbox"]

        # Clamp coordinates to image boundaries
        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(w, x2)
        y2_c = min(h, y2)

        # Ensure that y2_c is greater than y1_c
        if y2_c <= y1_c:
            y2_c = y1_c + 1

        # Ensure that x2_c is greater than x1_c
        if x2_c <= x1_c:
            x2_c = x1_c + 1

        cropped = final_img[y1_c:y2_c, x1_c:x2_c]
        out_name = f"{base_name}_page{page_number}_scan{seg_id}.png"
        out_path = os.path.join(EXPORT_DIR, out_name)
        cv2.imwrite(out_path, cropped)

def process_pdfs(docs_dir):
    pdf_files = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        page_images = pdf_to_images(pdf_path, dpi=DPI)

        for page_index, img in enumerate(page_images):
            page_number = page_index + 1
            # First, try to find the best angle
            # We do a rough angle search on the original image
            best_angle = find_best_angle(img)

            # Rotate the original image by best_angle
            best_rotated = rotate_image(img, best_angle)

            # Preprocess and extract final line boxes
            processed = preprocess(best_rotated)
            final_line_boxes = extract_lines_with_hierarchy(processed)

            if not final_line_boxes:
                # No lines, skip exporting
                continue

            segments = group_lines_into_segments(final_line_boxes, LINES_PER_SEGMENT)

            # Export each segment as an image file
            export_segment_images(pdf_name, page_number, best_rotated, segments)

if __name__ == "__main__":
    process_pdfs(DOCS_DIR)
    print("Processing complete. Check the 'export' folder for results.")

```

# main.py

```python
import customtkinter as ctk
from app import App

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()
```

# segment.py

```python
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

```

# app.py

```python
from managers.pdf_manager import PDFManager
from managers.segment_manager import SegmentManager
from utils.file_utils import *
from utils.config import *

import os
import math
import fitz  # PyMuPDF
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Historical Document Segmenter")
        self.geometry("1200x900")

        self.storage_data = load_storage()

        self.pdf_manager = PDFManager()
        self.segment_manager = SegmentManager()

        self.current_scan_page_number = None

        self.current_mode = "add"         # "add" or "edit"
        self.segment_input_mode = "drag"  # "drag" or "click"

        self.selected_segment_id = None
        self.selected_vertex_index = None
        self.selected_segment_scan_page = None
        self.rotation_angle = 0.0

        self.original_image = None
        self.current_image = None
        self.current_tkimage = None

        self.click_points = []

        # ---- TOP FRAME ----
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(side="top", fill="x", pady=5)

        # Left side labels for info
        self.pdf_name_label = ctk.CTkLabel(top_frame, text="PDF: ", width=200)
        self.pdf_name_label.pack(side="left", padx=10)

        self.page_number_label = ctk.CTkLabel(top_frame, text="Page: ", width=100)
        self.page_number_label.pack(side="left", padx=10)

        self.scan_page_label = ctk.CTkLabel(top_frame, text=f"Current Scan Page: ")
        self.scan_page_label.pack(side="left", padx=10)

        self.mode_label = ctk.CTkLabel(top_frame, text=f"Mode: {self.current_mode}")
        self.mode_label.pack(side="left", padx=10)

        self.segment_input_mode_label = ctk.CTkLabel(top_frame, text=f"Segment Input: {self.segment_input_mode}")
        self.segment_input_mode_label.pack(side="left", padx=10)

        self.rotation_scale = ctk.CTkSlider(
            top_frame, from_=0, to=360, number_of_steps=720, command=self.on_rotation_scale
        )
        self.rotation_scale.set(self.rotation_angle)
        self.rotation_scale.pack(side="left", padx=10)
        self.rotation_label = ctk.CTkLabel(top_frame, text=f"Rotation: {self.rotation_angle}°")
        self.rotation_label.pack(side="left", padx=10)

        # On the top frame, add the segment manipulation buttons
        top_btn_frame = ctk.CTkFrame(top_frame)
        top_btn_frame.pack(side="right", fill="x", padx=10)

        self.remove_last_segment_btn = ctk.CTkButton(top_btn_frame, text="Remove Last Segment (Shift)", command=self.remove_last_segment)
        self.remove_last_segment_btn.pack(side="left", padx=5)

        self.split_last_segment_btn = ctk.CTkButton(top_btn_frame, text="Split Last Segment (Tab)", command=self.split_last_segment)
        self.split_last_segment_btn.pack(side="left", padx=5)

        self.clear_segments_btn = ctk.CTkButton(top_btn_frame, text="Clear Segments (C)", command=self.clear_segments)
        self.clear_segments_btn.pack(side="left", padx=5)

        self.switch_mode_btn = ctk.CTkButton(top_btn_frame, text="Switch Add/Edit Mode (Ctrl)", command=self.switch_mode)
        self.switch_mode_btn.pack(side="left", padx=5)

        self.toggle_input_mode_btn = ctk.CTkButton(top_btn_frame, text="Toggle Drag/Click (D)", command=self.toggle_segment_input_mode)
        self.toggle_input_mode_btn.pack(side="left", padx=5)

        self.add_scan_page_btn = ctk.CTkButton(top_btn_frame, text="Add Scan Page (Alt)", command=self.add_scan_page)
        self.add_scan_page_btn.pack(side="left", padx=5)

        # ---- CANVAS ----
        self.canvas = tk.Canvas(self, bg="gray", width=1000, height=700)
        self.canvas.pack(expand=True, fill="both")
        self.canvas.bind("<Configure>", lambda e: self.update_canvas_image())

        # ---- BOTTOM FRAME ----
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(side="bottom", fill="x", pady=5)

        bottom_left_frame = ctk.CTkFrame(bottom_frame)
        bottom_left_frame.pack(side="left", padx=10)

        self.prev_page_btn = ctk.CTkButton(bottom_left_frame, text="Prev Page (A)", command=self.prev_page)
        self.prev_page_btn.pack(side="left", padx=5)

        self.next_page_btn = ctk.CTkButton(bottom_left_frame, text="Next Page (S)", command=self.next_page)
        self.next_page_btn.pack(side="left", padx=5)

        bottom_right_frame = ctk.CTkFrame(bottom_frame)
        bottom_right_frame.pack(side="right", padx=10)

        self.export_only_btn = ctk.CTkButton(bottom_right_frame, text="Export Segments (E)", command=self.export_segments_only)
        self.export_only_btn.pack(side="left", padx=5)

        self.export_page_btn = ctk.CTkButton(bottom_right_frame, text="Export Segments & Next (R)", command=self.export_and_next)
        self.export_page_btn.pack(side="right", padx=5)

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_left_button_press)
        self.canvas.bind("<B1-Motion>", self.on_left_button_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_button_release)

        self.drag_start = None
        self.drag_current = None
        self.is_dragging_vertex = False

        self.bind_keybindings()
        self.load_page_image()

    def bind_keybindings(self):
        # Segment editing
        self.bind_all('<Shift_L>', lambda e: self.remove_last_segment())
        self.bind_all('<Shift_R>', lambda e: self.remove_last_segment())
        self.bind_all('<c>', lambda e: self.clear_segments())
        self.bind_all('<C>', lambda e: self.clear_segments())
        self.bind_all('<Tab>', lambda e: self.split_last_segment())
        self.bind_all('<Control_L>', lambda e: self.switch_mode())
        self.bind_all('<Control_R>', lambda e: self.switch_mode())
        self.bind_all('<d>', lambda e: self.toggle_segment_input_mode())
        self.bind_all('<D>', lambda e: self.toggle_segment_input_mode())

        # Scan page changes
        self.bind_all('<z>', lambda e: self.change_scan_page(-1))
        self.bind_all('<Z>', lambda e: self.change_scan_page(-1))
        self.bind_all('<x>', lambda e: self.change_scan_page(1))
        self.bind_all('<X>', lambda e: self.change_scan_page(1))
        self.bind_all('<Alt_L>', lambda e: self.add_scan_page())
        self.bind_all('<Alt_R>', lambda e: self.add_scan_page())

        # Page navigation
        self.bind_all('<a>', lambda e: self.prev_page())
        self.bind_all('<A>', lambda e: self.prev_page())
        self.bind_all('<s>', lambda e: self.next_page())
        self.bind_all('<S>', lambda e: self.next_page())

        # Rotation
        self.bind_all('<Left>', lambda e: self.rotate_image(-0.5))
        self.bind_all('<Right>', lambda e: self.rotate_image(0.5))

        # Export
        self.bind_all('<e>', lambda e: self.export_segments_only())
        self.bind_all('<E>', lambda e: self.export_segments_only())
        self.bind_all('<r>', lambda e: self.export_and_next())
        self.bind_all('<R>', lambda e: self.export_and_next())

    def load_page_image(self):
        page = self.pdf_manager.get_current_page()
        if page is None:
            messagebox.showwarning("Warning", "No page available to load.")
            return

        pdf_path = self.pdf_manager.get_current_pdf_path()
        pdf_hash = compute_pdf_hash(pdf_path)
        pdf_page_number = self.pdf_manager.get_current_page_index() + 1

        pdf_entry = self.storage_data["pdfs"].get(pdf_hash, None)
        if pdf_entry:
            page_entry = pdf_entry.get("pages", {}).get(str(pdf_page_number), None)
            if page_entry:
                self.rotation_angle = page_entry.get("rotation_angle", 0.0)
                scan_pages_data = page_entry.get("scan_pages", [])
                segments = []
                for sp in scan_pages_data:
                    sp_num = sp["scan_page_number"]
                    for seg in sp["segments"]:
                        seg_points = [[float(x), float(y)] for (x, y) in seg['original_points']]
                        segments.append({
                            'original_points': seg_points,
                            'id': seg['id'],
                            'scan_page': sp_num
                        })
                self.segment_manager.set_segments(segments)

                existing_scan_pages = self.segment_manager.get_scan_pages()
                if existing_scan_pages:
                    # Set current scan page to the highest existing plus 1 logic doesn't apply here.
                    # Instead, we just pick the last one used (or user can add new page)
                    self.current_scan_page_number = existing_scan_pages[-1]
                else:
                    self.add_scan_page()
            else:
                self.rotation_angle = 0.0
                self.segment_manager.clear()
                self.add_scan_page()
        else:
            self.rotation_angle = 0.0
            self.segment_manager.clear()
            self.add_scan_page()

        zoom_x = 4.0
        zoom_y = 4.0
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        self.original_image = img

        self.current_image = self.original_image.rotate(-self.rotation_angle, expand=True)

        self.rotation_scale.set(self.rotation_angle)
        self.rotation_label.configure(text=f"Rotation: {self.rotation_angle}°")

        self.update_canvas_image()
        self.update_labels()

    def update_canvas_image(self):
        if self.current_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 10: canvas_width = 1000
        if canvas_height < 10: canvas_height = 700

        img_w, img_h = self.current_image.size
        scale = min(canvas_width / img_w, canvas_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.current_tkimage = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.current_tkimage, anchor="center")

        self.draw_segments()

        if self.segment_input_mode == "click" and self.click_points:
            rotated_points = self.get_rotated_points(self.click_points)
            canvas_points = self.image_points_to_canvas(rotated_points)
            for (cx, cy) in canvas_points:
                self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5, outline="green", width=2)

    def update_labels(self):
        pdf_name = self.pdf_manager.get_current_pdf_name()
        pdf_page_idx = self.pdf_manager.get_current_page_index() + 1
        pdf_page_count = self.pdf_manager.get_pdf_page_count()

        if pdf_name is None:
            self.pdf_name_label.configure(text="PDF: None")
        else:
            self.pdf_name_label.configure(text=f"PDF: {pdf_name}")

        self.page_number_label.configure(text=f"Page: {pdf_page_idx}/{pdf_page_count}")

        if self.current_scan_page_number is not None:
            self.scan_page_label.configure(text=f"Current Scan Page: {self.current_scan_page_number}")
        else:
            self.scan_page_label.configure(text="Current Scan Page: None")

        self.mode_label.configure(text=f"Mode: {self.current_mode}")
        self.segment_input_mode_label.configure(text=f"Segment Input: {self.segment_input_mode}")

    def draw_segments(self):
        for seg in self.segment_manager.get_segments():
            original_points = seg['original_points']
            rotated_points = self.get_rotated_points(original_points)
            canvas_points = self.image_points_to_canvas(rotated_points)
            self.canvas.create_polygon(
                canvas_points,
                fill="#888888",
                outline="white",
                width=2,
                stipple="gray50"
            )
            cx = sum([p[0] for p in canvas_points]) / len(canvas_points)
            cy = sum([p[1] for p in canvas_points]) / len(canvas_points)
            label_text = f"Segment {seg['id']} - Page {seg['scan_page']}"
            self.canvas.create_text(cx, cy, text=label_text, fill="white", font=("Arial", 14, "bold"))

            if self.current_mode == "edit":
                for (vx, vy) in canvas_points:
                    self.canvas.create_rectangle(vx-5, vy-5, vx+5, vy+5, outline="red", width=2)

    def get_rotated_points(self, points):
        if not self.original_image:
            return points

        orig_w, orig_h = self.original_image.size
        orig_cx, orig_cy = orig_w / 2, orig_h / 2

        rotated = []
        theta = math.radians(self.rotation_angle)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        rotated_w, rotated_h = self.current_image.size
        new_cx, new_cy = rotated_w / 2, rotated_h / 2

        for (x, y) in points:
            x_shifted = x - orig_cx
            y_shifted = y - orig_cy
            x_rot = x_shifted * cos_theta - y_shifted * sin_theta
            y_rot = x_shifted * sin_theta + y_shifted * cos_theta
            x_new = x_rot + new_cx
            y_new = y_rot + new_cy
            rotated.append([x_new, y_new])
        return rotated

    def reverse_rotate_points(self, rotated_points):
        if not self.original_image:
            return rotated_points

        orig_w, orig_h = self.original_image.size
        orig_cx, orig_cy = orig_w / 2, orig_h / 2

        rotated_w, rotated_h = self.current_image.size
        new_cx, new_cy = rotated_w / 2, rotated_h / 2

        theta = math.radians(-self.rotation_angle)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        original_points = []
        for (x, y) in rotated_points:
            x_shifted = x - new_cx
            y_shifted = y - new_cy
            x_orig = x_shifted * cos_theta - y_shifted * sin_theta + orig_cx
            y_orig = x_shifted * sin_theta + y_shifted * cos_theta + orig_cy
            original_points.append([x_orig, y_orig])
        return original_points

    def image_points_to_canvas(self, points):
        if self.current_tkimage is None:
            return points
        img_w, img_h = self.current_image.size
        tk_w = self.current_tkimage.width()
        tk_h = self.current_tkimage.height()

        scale_x = tk_w / img_w
        scale_y = tk_h / img_h

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        offset_x = (canvas_w - tk_w) / 2
        offset_y = (canvas_h - tk_h) / 2

        canvas_points = []
        for (x, y) in points:
            cx = x * scale_x + offset_x
            cy = y * scale_y + offset_y
            canvas_points.append((cx, cy))
        return canvas_points

    def canvas_points_to_image(self, points):
        img_w, img_h = self.current_image.size
        tk_w = self.current_tkimage.width()
        tk_h = self.current_tkimage.height()

        scale_x = tk_w / img_w
        scale_y = tk_h / img_h

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        offset_x = (canvas_w - tk_w) / 2
        offset_y = (canvas_h - tk_h) / 2

        image_points = []
        for (cx, cy) in points:
            x = (cx - offset_x) / scale_x
            y = (cy - offset_y) / scale_y
            image_points.append((x, y))
        return image_points

    def on_left_button_press(self, event):
        x, y = event.x, event.y
        if self.current_mode == "add":
            if self.segment_input_mode == "drag":
                self.drag_start = (x, y)
                self.drag_current = (x, y)
            else:
                clicked_pt_image = self.canvas_points_to_image([(x, y)])[0]
                original_pt = self.reverse_rotate_points([clicked_pt_image])[0]
                self.click_points.append(original_pt)
                self.update_canvas_image()
                if len(self.click_points) == 4:
                    ordered = self.order_points(self.get_rotated_points(self.click_points))
                    original_quad = self.reverse_rotate_points(ordered)
                    if self.current_scan_page_number is None:
                        self.add_scan_page()
                    self.segment_manager.add_segment(original_quad, self.current_scan_page_number)
                    self.click_points.clear()
                    self.update_canvas_image()

        elif self.current_mode == "edit":
            clicked_seg_id, vertex_index, scan_page = self.find_nearest_vertex(x, y)
            if clicked_seg_id is not None:
                self.selected_segment_id = clicked_seg_id
                self.selected_vertex_index = vertex_index
                self.selected_segment_scan_page = scan_page
                self.is_dragging_vertex = True

    def on_left_button_move(self, event):
        x, y = event.x, event.y
        if self.current_mode == "add" and self.segment_input_mode == "drag" and self.drag_start is not None:
            self.drag_current = (x, y)
            self.update_canvas_image()
            sx, sy = self.drag_start
            self.canvas.create_rectangle(sx, sy, x, y, outline="red", width=2)
        elif self.current_mode == "edit" and self.is_dragging_vertex:
            seg = self.get_segment_by_id(self.selected_segment_id, self.selected_segment_scan_page)
            if seg:
                original_points = seg['original_points']
                rotated_points = self.get_rotated_points(original_points)
                rotated_pt = self.canvas_points_to_image([(x, y)])
                new_x, new_y = rotated_pt[0]
                rotated_points[self.selected_vertex_index] = [new_x, new_y]
                new_original_points = self.reverse_rotate_points(rotated_points)
                self.segment_manager.update_segment_points(seg['id'], seg['scan_page'], new_original_points)
                self.update_canvas_image()

    def on_left_button_release(self, event):
        if self.current_mode == "add" and self.segment_input_mode == "drag":
            if self.drag_start is not None and self.drag_current is not None:
                sx, sy = self.drag_start
                ex, ey = self.drag_current
                img_points = self.canvas_points_to_image([(sx, sy), (ex, sy), (ex, ey), (sx, ey)])
                original_points = self.reverse_rotate_points(img_points)
                if self.current_scan_page_number is None:
                    self.add_scan_page()
                self.segment_manager.add_segment(original_points, self.current_scan_page_number)
                self.update_canvas_image()
            self.drag_start = None
            self.drag_current = None
        elif self.current_mode == "edit":
            if self.is_dragging_vertex:
                self.is_dragging_vertex = False
                self.selected_segment_id = None
                self.selected_vertex_index = None
                self.selected_segment_scan_page = None
                self.update_canvas_image()

    def find_nearest_vertex(self, cx, cy, threshold=10):
        for seg in self.segment_manager.get_segments():
            original_points = seg['original_points']
            rotated_points = self.get_rotated_points(original_points)
            canvas_rotated_points = self.image_points_to_canvas(rotated_points)
            for i, (vx, vy) in enumerate(canvas_rotated_points):
                dist = math.dist((cx, cy), (vx, vy))
                if dist < threshold:
                    return seg['id'], i, seg['scan_page']
        return None, None, None

    def get_segment_by_id(self, seg_id, scan_page_number):
        for seg in self.segment_manager.get_segments():
            if seg['id'] == seg_id and seg['scan_page'] == scan_page_number:
                return seg
        return None

    def next_page(self):
        self.save_current_page_to_storage()
        if self.pdf_manager.next_page():
            self.segment_manager.clear()
            self.current_scan_page_number = None
            self.rotation_angle = 0.0
            self.update_rotation_scale()
            self.load_page_image()
        else:
            if self.pdf_manager.next_pdf():
                self.segment_manager.clear()
                self.current_scan_page_number = None
                self.rotation_angle = 0.0
                self.update_rotation_scale()
                self.load_page_image()
            else:
                messagebox.showinfo("Info", "No more PDFs available.")

    def prev_page(self):
        self.save_current_page_to_storage()
        if self.pdf_manager.prev_page():
            self.segment_manager.clear()
            self.current_scan_page_number = None
            self.rotation_angle = 0.0
            self.update_rotation_scale()
            self.load_page_image()
        else:
            if self.pdf_manager.prev_pdf():
                self.segment_manager.clear()
                self.current_scan_page_number = None
                self.rotation_angle = 0.0
                self.update_rotation_scale()
                self.load_page_image()
            else:
                messagebox.showinfo("Info", "No previous PDF/page available.")

    def remove_last_segment(self):
        if not self.segment_manager.get_segments():
            messagebox.showwarning("Warning", "No segments to remove.")
            return
        self.segment_manager.remove_last_segment()
        self.update_canvas_image()

    def split_last_segment(self):
        segments = self.segment_manager.get_segments()
        if not segments:
            messagebox.showwarning("Warning", "No segments to split.")
            return

        last_seg = segments[-1]
        original_points = last_seg['original_points']
        scan_page = last_seg['scan_page']

        rotated_points = self.get_rotated_points(original_points)
        ordered = self.order_points(rotated_points)
        (tl, tr, br, bl) = ordered

        N = 5
        new_segments = []
        for i in range(N):
            t_ratio1 = i / N
            t_ratio2 = (i+1) / N

            top_left = [tl[0] + (bl[0] - tl[0])*t_ratio1, tl[1] + (bl[1] - tl[1])*t_ratio1]
            top_right = [tr[0] + (br[0] - tr[0])*t_ratio1, tr[1] + (br[1] - tr[1])*t_ratio1]

            bottom_left = [tl[0] + (bl[0] - tl[0])*t_ratio2, tl[1] + (bl[1] - tl[1])*t_ratio2]
            bottom_right = [tr[0] + (br[0] - tr[0])*t_ratio2, tr[1] + (br[1] - tr[1])*t_ratio2]

            sub_quad = [top_left, top_right, bottom_right, bottom_left]
            original_sub_quad = self.reverse_rotate_points(sub_quad)
            original_sub_quad = [[float(x), float(y)] for (x, y) in original_sub_quad]

            new_segments.append(original_sub_quad)

        self.segment_manager.remove_last_segment()

        for seg_points in new_segments:
            self.segment_manager.add_segment(seg_points, scan_page)

        self.update_canvas_image()

    def clear_segments(self):
        if not self.segment_manager.get_segments():
            messagebox.showwarning("Warning", "No segments to clear.")
            return
        self.segment_manager.clear()
        self.current_scan_page_number = None
        self.add_scan_page()
        self.update_canvas_image()

    def add_scan_page(self):
        # Determine next scan page by looking at all existing pages
        existing_pages = self.segment_manager.get_scan_pages()
        if existing_pages:
            self.current_scan_page_number = max(existing_pages) + 1
        else:
            self.current_scan_page_number = 1
        self.update_labels()

    def change_scan_page(self, direction):
        if self.current_scan_page_number is None:
            self.add_scan_page()
            return

        new_page = self.current_scan_page_number + direction
        if new_page < 1:
            return

        # If the page doesn't exist yet, we can still navigate to it.
        # If it's beyond existing pages, it's a new page number that user can define segments for.
        # Just set to new_page.
        self.current_scan_page_number = new_page
        self.update_labels()

    def switch_mode(self):
        if self.current_mode == "add":
            self.current_mode = "edit"
        else:
            self.current_mode = "add"
        self.update_labels()
        self.update_canvas_image()

    def toggle_segment_input_mode(self):
        if self.segment_input_mode == "drag":
            self.segment_input_mode = "click"
            self.drag_start = None
            self.drag_current = None
        else:
            self.segment_input_mode = "drag"
            self.click_points.clear()
        self.update_labels()
        self.update_canvas_image()

    def export_segments_only(self):
        pdf_name = self.pdf_manager.get_current_pdf_name()
        if pdf_name is None:
            messagebox.showwarning("Warning", "No PDF loaded.")
            return
        base_name = os.path.splitext(pdf_name)[0]
        segments = self.segment_manager.get_segments()

        if not segments:
            messagebox.showwarning("Warning", "No segments to export.")
            return

        for seg in segments:
            sp_num = seg['scan_page']
            out_name = f"{base_name}_page{sp_num}_segment{seg['id']}.png"
            out_path = os.path.join(EXPORT_DIR, out_name)
            self.export_segment(seg['original_points'], out_path)

        self.save_current_page_to_storage()
        messagebox.showinfo("Info", "Segments exported successfully.")

    def export_and_next(self):
        pdf_name = self.pdf_manager.get_current_pdf_name()
        if pdf_name is None:
            messagebox.showwarning("Warning", "No PDF loaded.")
            return
        base_name = os.path.splitext(pdf_name)[0]
        segments = self.segment_manager.get_segments()

        if not segments:
            messagebox.showwarning("Warning", "No segments to export.")
            return

        for seg in segments:
            sp_num = seg['scan_page']
            out_name = f"{base_name}_page{sp_num}_segment{seg['id']}.png"
            out_path = os.path.join(EXPORT_DIR, out_name)
            self.export_segment(seg['original_points'], out_path)

        self.save_current_page_to_storage()

        if self.pdf_manager.next_page():
            self.segment_manager.clear()
            self.current_scan_page_number = None
            self.rotation_angle = 0.0
            self.update_rotation_scale()
            self.load_page_image()
            messagebox.showinfo("Info", "Page exported successfully. Proceeding to next PDF page.")
        else:
            if self.pdf_manager.next_pdf():
                self.segment_manager.clear()
                self.current_scan_page_number = None
                self.rotation_angle = 0.0
                self.update_rotation_scale()
                self.load_page_image()
                messagebox.showinfo("Info", "Page exported successfully. Proceeding to next PDF page.")
            else:
                messagebox.showinfo("Info", "Page exported successfully. No more PDFs/pages available.")

    def export_segment(self, points, out_path):
        rotated_points = self.get_rotated_points(points)
        if len(rotated_points) != 4:
            messagebox.showwarning("Warning", f"Segment does not have 4 points. Skipping export for {out_path}.")
            return

        ordered = self.order_points(rotated_points)
        (tl, tr, br, bl) = ordered

        def dist(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        widthA = dist(br, bl)
        widthB = dist(tr, tl)
        maxWidth = int(max(widthA, widthB))

        heightA = dist(tr, br)
        heightB = dist(tl, bl)
        maxHeight = int(max(heightA, heightB))

        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([
            [0, 0],
            [maxWidth-1, 0],
            [maxWidth-1, maxHeight-1],
            [0, maxHeight-1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)

        cv_img = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
        warped = cv2.warpPerspective(cv_img, M, (maxWidth, maxHeight))
        cv2.imwrite(out_path, warped)

    def order_points(self, pts):
        pts = np.array(pts, dtype="float32")
        y_sorted = pts[np.argsort(pts[:,1]), :]

        top = y_sorted[:2, :]
        bottom = y_sorted[2:, :]

        top = top[np.argsort(top[:,0]), :]
        tl, tr = top[0], top[1]

        bottom = bottom[np.argsort(bottom[:,0]), :]
        bl, br = bottom[0], bottom[1]

        return [tl, tr, br, bl]

    def rotate_image(self, angle_increment):
        self.rotation_angle = (self.rotation_angle + angle_increment) % 360
        if angle_increment != 0:
            self.rotation_scale.set(self.rotation_angle)
        self.rotation_label.configure(text=f"Rotation: {self.rotation_angle}°")
        self.current_image = self.original_image.rotate(-self.rotation_angle, expand=True)
        self.update_canvas_image()

    def update_rotation_scale(self):
        self.rotation_scale.set(self.rotation_angle)
        self.rotation_label.configure(text=f"Rotation: {self.rotation_angle}°")

    def on_rotation_scale(self, value):
        desired_angle = int(float(value))
        angle_increment = desired_angle - self.rotation_angle
        self.rotate_image(angle_increment)

    def save_current_page_to_storage(self):
        pdf_path = self.pdf_manager.get_current_pdf_path()
        if pdf_path is None:
            return
        pdf_hash = compute_pdf_hash(pdf_path)
        pdf_entry = self.storage_data["pdfs"].get(pdf_hash, {})
        pdf_entry["pdf_path"] = pdf_path

        page_number = self.pdf_manager.get_current_page_index() + 1
        segments = self.segment_manager.get_segments()

        scan_pages = {}
        for seg in segments:
            sp_num = seg['scan_page']
            cleaned_points = [[float(x), float(y)] for (x, y) in seg['original_points']]
            if sp_num not in scan_pages:
                scan_pages[sp_num] = []
            scan_pages[sp_num].append({
                'original_points': cleaned_points,
                'id': seg['id']
            })

        scan_pages_list = []
        for sp_num in sorted(scan_pages.keys()):
            scan_pages_list.append({
                "scan_page_number": sp_num,
                "segments": scan_pages[sp_num]
            })

        if "pages" not in pdf_entry:
            pdf_entry["pages"] = {}
        page_entry = {}
        page_entry["rotation_angle"] = float(self.rotation_angle)
        page_entry["scan_pages"] = scan_pages_list
        pdf_entry["pages"][str(page_number)] = page_entry

        self.storage_data["pdfs"][pdf_hash] = pdf_entry
        save_storage(self.storage_data)

```

# transcribe.py

```python
import os
import base64
import requests
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
import re
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION VARIABLES
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
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

```

# file_utils.py

```python
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

```

# config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"
EXPORT_DIR = "export"
STORAGE_FILE = "segments.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

```

# pdf_manager.py

```python
import glob
import os
import fitz

from utils.file_utils import natural_sort_key
from utils.config import DOCS_DIR

class PDFManager:
    def __init__(self, docs_dir=DOCS_DIR):
        self.pdf_files = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")), key=natural_sort_key)
        self.current_pdf_index = 0
        self.current_page_index = 0
        self.current_doc = None
        if self.pdf_files:
            self.load_pdf(self.pdf_files[self.current_pdf_index])
        else:
            self.current_doc = None

    def load_pdf(self, pdf_path):
        if self.current_doc:
            self.current_doc.close()
        self.current_doc = fitz.open(pdf_path)
        self.current_page_index = 0

    def get_current_pdf_name(self):
        if not self.pdf_files:
            return None
        return os.path.basename(self.pdf_files[self.current_pdf_index])

    def get_total_pdfs(self):
        return len(self.pdf_files)

    def get_pdf_page_count(self):
        if self.current_doc:
            return self.current_doc.page_count
        return 0

    def get_current_page(self):
        if self.current_doc and 0 <= self.current_page_index < self.current_doc.page_count:
            return self.current_doc.load_page(self.current_page_index)
        return None

    def next_page(self):
        if self.current_doc:
            if self.current_page_index < self.current_doc.page_count - 1:
                self.current_page_index += 1
                return True
            else:
                return False
        return False

    def prev_page(self):
        if self.current_doc:
            if self.current_page_index > 0:
                self.current_page_index -= 1
                return True
            else:
                return False
        return False

    def next_pdf(self):
        if self.current_pdf_index < len(self.pdf_files)-1:
            self.current_pdf_index += 1
            self.load_pdf(self.pdf_files[self.current_pdf_index])
            return True
        return False

    def prev_pdf(self):
        if self.current_pdf_index > 0:
            self.current_pdf_index -= 1
            self.load_pdf(self.pdf_files[self.current_pdf_index])
            return True
        return False

    def get_current_pdf_index(self):
        return self.current_pdf_index

    def get_current_page_index(self):
        return self.current_page_index

    def get_current_pdf_path(self):
        if self.pdf_files:
            return self.pdf_files[self.current_pdf_index]
        return None

```

# segment_manager.py

```python
class SegmentManager:
    def __init__(self):
        self.segments = []

    def add_segment(self, points, scan_page_number):
        points = [[float(px), float(py)] for (px, py) in points]
        existing = [s for s in self.segments if s['scan_page'] == scan_page_number]
        seg_id = len(existing) + 1
        self.segments.append({
            'original_points': points,
            'id': seg_id,
            'scan_page': scan_page_number
        })

    def remove_last_segment(self):
        if self.segments:
            self.segments.pop()

    def clear(self):
        self.segments.clear()

    def get_segments(self):
        return self.segments

    def set_segments(self, segments):
        for seg in segments:
            seg['original_points'] = [[float(x), float(y)] for (x, y) in seg['original_points']]
        self.segments = segments

    def update_segment_points(self, seg_id, scan_page_number, new_points):
        new_points = [[float(px), float(py)] for (px, py) in new_points]
        for seg in self.segments:
            if seg['id'] == seg_id and seg['scan_page'] == scan_page_number:
                seg['original_points'] = new_points
                break

    def get_scan_pages(self):
        pages = set(s['scan_page'] for s in self.segments)
        return sorted(list(pages))

    def get_segments_by_scan_page(self, scan_page_number):
        return [s for s in self.segments if s['scan_page'] == scan_page_number]

```

