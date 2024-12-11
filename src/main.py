import os
import glob
import math
import fitz  # PyMuPDF
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
import json
import hashlib
import re

# Ensure customtkinter is using a modern theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

DOCS_DIR = "docs"
EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

STORAGE_FILE = "segments.json"

def compute_pdf_hash(pdf_path):
    hash_sha256 = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def natural_sort_key(s):
    base = os.path.basename(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', base)]

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

if __name__ == "__main__":
    app = App()
    app.canvas.bind("<Configure>", lambda e: app.update_canvas_image())
    app.mainloop()
