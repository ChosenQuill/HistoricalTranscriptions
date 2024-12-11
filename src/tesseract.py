# Experimental proof of concept script where we try using pytesseract to create segments, not used in final application. Uses terreract v5. 

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
