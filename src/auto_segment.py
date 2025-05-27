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
