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
from customtkinter import filedialog as fd
import json
import base64
import requests
import re
from pathlib import Path
from collections import defaultdict
from typing import List
import threading

class ProjectManager:
    def __init__(self):
        self.project_path = None
        self.project_data = None
        self.project_json_path = None

    def new_project(self, parent):
        project_dir = fd.askdirectory(title="Select Project Directory", parent=parent)
        if not project_dir:
            return False
        pdf_files = fd.askopenfilenames(title="Select PDF Files for Project", filetypes=[("PDF Files", "*.pdf")], parent=parent)
        if not pdf_files:
            return False
        # Copy PDFs into project folder
        pdfs_dir = os.path.join(project_dir, "pdfs")
        os.makedirs(pdfs_dir, exist_ok=True)
        pdf_names = []
        for pdf in pdf_files:
            base = os.path.basename(pdf)
            dest = os.path.join(pdfs_dir, base)
            if not os.path.exists(dest):
                with open(pdf, "rb") as fsrc, open(dest, "wb") as fdst:
                    fdst.write(fsrc.read())
            pdf_names.append(base)
        # Create segments and transcripts folders
        os.makedirs(os.path.join(project_dir, "segments"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "transcripts"), exist_ok=True)
        # Create project.json
        self.project_json_path = os.path.join(project_dir, "project.json")
        self.project_data = {
            "pdfs": pdf_names,
            "segments": {},
            "transcripts": {},
        }
        with open(self.project_json_path, "w") as f:
            json.dump(self.project_data, f, indent=2)
        self.project_path = project_dir
        return True

    def open_project(self, parent):
        project_dir = fd.askdirectory(title="Open Project Directory", parent=parent)
        if not project_dir:
            return False
        project_json = os.path.join(project_dir, "project.json")
        if not os.path.exists(project_json):
            tk.messagebox.showerror("Error", "No project.json found in selected directory.")
            return False
        with open(project_json, "r") as f:
            self.project_data = json.load(f)
        self.project_path = project_dir
        self.project_json_path = project_json
        return True

    def save_project(self):
        if self.project_json_path and self.project_data:
            with open(self.project_json_path, "w") as f:
                json.dump(self.project_data, f, indent=2)

    def new_project_with_paths(self, parent, project_dir, pdf_files):
        if not project_dir or not pdf_files:
            return False
        pdfs_dir = os.path.join(project_dir, "pdfs")
        os.makedirs(pdfs_dir, exist_ok=True)
        pdf_names = []
        for pdf in pdf_files:
            base = os.path.basename(pdf)
            dest = os.path.join(pdfs_dir, base)
            if not os.path.exists(dest):
                with open(pdf, "rb") as fsrc, open(dest, "wb") as fdst:
                    fdst.write(fsrc.read())
            pdf_names.append(base)
        os.makedirs(os.path.join(project_dir, "segments"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "transcripts"), exist_ok=True)
        self.project_json_path = os.path.join(project_dir, "project.json")
        self.project_data = {
            "pdfs": pdf_names,
            "segments": {},
            "transcripts": {},
        }
        with open(self.project_json_path, "w") as f:
            json.dump(self.project_data, f, indent=2)
        self.project_path = project_dir
        return True

    def open_project_with_path(self, parent, project_dir):
        project_json = os.path.join(project_dir, "project.json")
        if not os.path.exists(project_json):
            tk.messagebox.showerror("Error", "No project.json found in selected directory.")
            return False
        with open(project_json, "r") as f:
            self.project_data = json.load(f)
        self.project_path = project_dir
        self.project_json_path = project_json
        return True

class StartWindow(ctk.CTkToplevel):
    def __init__(self, master, on_project_selected):
        super().__init__(master)
        self.title("Start Project")
        self.geometry("400x220")
        self.on_project_selected = on_project_selected
        self.pm = ProjectManager()
        self.configure(bg="#181a1b")
        # Load icons for start window
        icon_dir = os.path.join(os.path.dirname(__file__), "icons")
        self.icon_new = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "new.png")), size=(24, 24))
        self.icon_open = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "open.png")), size=(24, 24))
        label = ctk.CTkLabel(self, text="Welcome! Start a new project or open an existing one.", font=("Arial", 16, "bold"))
        label.pack(pady=24)
        btn_new = ctk.CTkButton(self, image=self.icon_new, text="New Project", compound="left", command=self.new_project, width=180, font=("Arial", 14, "bold"))
        btn_new.pack(pady=10)
        btn_open = ctk.CTkButton(self, image=self.icon_open, text="Open Project", compound="left", command=self.open_project, width=180, font=("Arial", 14, "bold"))
        btn_open.pack(pady=10)

    def new_project(self):
        # Use a custom ctk dialog for folder selection if possible
        project_dir = self.ask_directory("Select Project Directory")
        if not project_dir:
            return False
        pdf_files = self.ask_open_files("Select PDF Files for Project", filetypes=[("PDF Files", "*.pdf")])
        if not pdf_files:
            return False
        if self.pm.new_project_with_paths(self, project_dir, pdf_files):
            self.on_project_selected(self.pm)
            self.destroy()

    def open_project(self):
        project_dir = self.ask_directory("Open Project Directory")
        if not project_dir:
            return False
        if self.pm.open_project_with_path(self, project_dir):
            self.on_project_selected(self.pm)
            self.destroy()

    def ask_directory(self, title):
        # CustomTkinter does not have a native directory picker, so fallback to tk.filedialog but theme the parent
        import tkinter.filedialog as fd
        return fd.askdirectory(title=title, parent=self)

    def ask_open_files(self, title, filetypes):
        import tkinter.filedialog as fd
        return fd.askopenfilenames(title=title, filetypes=filetypes, parent=self)

# Refactor App to use StartWindow and ProjectManager
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.title("Historical Document Segmenter")
        self.geometry("1200x900")
        self.project_manager = None
        self.wait_visibility()
        self.withdraw()
        def on_project_selected(pm):
            self.project_manager = pm
            self.deiconify()
            self.init_main_ui()
        StartWindow(self, on_project_selected)

    def init_main_ui(self):
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

        self.minsize(1200, 900)

        # Load icons
        icon_dir = os.path.join(os.path.dirname(__file__), "icons")
        self.icon_new = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "new.png")), size=(24, 24))
        self.icon_open = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "open.png")), size=(24, 24))
        self.icon_save = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "save.png")), size=(24, 24))
        self.icon_exit = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "exit.png")), size=(24, 24))
        self.icon_remove = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "remove.png")), size=(24, 24))
        self.icon_split = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "split.png")), size=(24, 24))
        self.icon_clear = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "clear.png")), size=(24, 24))
        self.icon_mode = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "mode.png")), size=(24, 24))
        self.icon_drag = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "drag.png")), size=(24, 24))
        self.icon_addpage = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "addpage.png")), size=(24, 24))
        self.icon_transcribe = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "transcribe.png")), size=(24, 24))
        self.icon_prev = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "prev.png")), size=(24, 24))
        self.icon_next = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "next.png")), size=(24, 24))
        self.icon_export = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "export.png")), size=(24, 24))
        self.icon_exportnext = ctk.CTkImage(light_image=Image.open(os.path.join(icon_dir, "exportnext.png")), size=(24, 24))

        # ---- PROJECT MENU FRAME (replaces tk.Menu) ----
        menu_frame = ctk.CTkFrame(self, corner_radius=10)
        menu_frame.pack(side="top", fill="x", pady=5, padx=5)
        self.menu_new_btn = ctk.CTkButton(menu_frame, image=self.icon_new, text="New Project", compound="left", command=self.menu_new_project, width=140, font=("Arial", 14, "bold"))
        self.menu_new_btn.pack(side="left", padx=8, pady=4)
        self.menu_open_btn = ctk.CTkButton(menu_frame, image=self.icon_open, text="Open Project", compound="left", command=self.menu_open_project, width=140, font=("Arial", 14, "bold"))
        self.menu_open_btn.pack(side="left", padx=8, pady=4)
        self.menu_save_btn = ctk.CTkButton(menu_frame, image=self.icon_save, text="Save Project", compound="left", command=self.menu_save_project, width=140, font=("Arial", 14, "bold"))
        self.menu_save_btn.pack(side="left", padx=8, pady=4)
        self.menu_exit_btn = ctk.CTkButton(menu_frame, image=self.icon_exit, text="Exit", compound="left", command=self.quit, width=100, font=("Arial", 14, "bold"))
        self.menu_exit_btn.pack(side="left", padx=8, pady=4)

        # ---- TOP FRAME ----
        top_frame = ctk.CTkFrame(self, corner_radius=10)
        top_frame.pack(side="top", fill="x", pady=5, padx=5)

        # Left side labels for info
        self.pdf_name_label = ctk.CTkLabel(top_frame, text="PDF: ", width=200, font=("Arial", 13, "bold"))
        self.pdf_name_label.pack(side="left", padx=10)

        self.page_number_label = ctk.CTkLabel(top_frame, text="Page: ", width=100, font=("Arial", 13, "bold"))
        self.page_number_label.pack(side="left", padx=10)

        self.scan_page_label = ctk.CTkLabel(top_frame, text=f"Current Scan Page: ", font=("Arial", 13, "bold"))
        self.scan_page_label.pack(side="left", padx=10)

        self.mode_label = ctk.CTkLabel(top_frame, text=f"Mode: {self.current_mode}", font=("Arial", 13, "bold"))
        self.mode_label.pack(side="left", padx=10)

        self.segment_input_mode_label = ctk.CTkLabel(top_frame, text=f"Segment Input: {self.segment_input_mode}", font=("Arial", 13, "bold"))
        self.segment_input_mode_label.pack(side="left", padx=10)

        self.rotation_scale = ctk.CTkSlider(
            top_frame, from_=0, to=360, number_of_steps=720, command=self.on_rotation_scale, width=200
        )
        self.rotation_scale.set(self.rotation_angle)
        self.rotation_scale.pack(side="left", padx=10)
        self.rotation_label = ctk.CTkLabel(top_frame, text=f"Rotation: {self.rotation_angle}°", font=("Arial", 13))
        self.rotation_label.pack(side="left", padx=10)

        # On the top frame, add the segment manipulation buttons
        top_btn_frame = ctk.CTkFrame(top_frame, corner_radius=10)
        top_btn_frame.pack(side="right", fill="x", padx=10)

        self.remove_last_segment_btn = ctk.CTkButton(top_btn_frame, image=self.icon_remove, text="Remove Last (Shift)", compound="left", command=self.remove_last_segment, width=120, font=("Arial", 12))
        self.remove_last_segment_btn.pack(side="left", padx=5)
        self.split_last_segment_btn = ctk.CTkButton(top_btn_frame, image=self.icon_split, text="Split Last (Tab)", compound="left", command=self.split_last_segment, width=120, font=("Arial", 12))
        self.split_last_segment_btn.pack(side="left", padx=5)
        self.clear_segments_btn = ctk.CTkButton(top_btn_frame, image=self.icon_clear, text="Clear (C)", compound="left", command=self.clear_segments, width=100, font=("Arial", 12))
        self.clear_segments_btn.pack(side="left", padx=5)
        self.switch_mode_btn = ctk.CTkButton(top_btn_frame, image=self.icon_mode, text="Add/Edit (Ctrl)", compound="left", command=self.switch_mode, width=120, font=("Arial", 12))
        self.switch_mode_btn.pack(side="left", padx=5)
        self.toggle_input_mode_btn = ctk.CTkButton(top_btn_frame, image=self.icon_drag, text="Drag/Click (D)", compound="left", command=self.toggle_segment_input_mode, width=120, font=("Arial", 12))
        self.toggle_input_mode_btn.pack(side="left", padx=5)
        self.add_scan_page_btn = ctk.CTkButton(top_btn_frame, image=self.icon_addpage, text="Add Scan Page (Alt)", compound="left", command=self.add_scan_page, width=140, font=("Arial", 12))
        self.add_scan_page_btn.pack(side="left", padx=5)
        self.transcribe_btn = ctk.CTkButton(top_btn_frame, image=self.icon_transcribe, text="Transcribe (T)", compound="left", command=self.transcribe_segments, width=140, font=("Arial", 12, "bold"))
        self.transcribe_btn.pack(side="left", padx=5)

        # ---- MAIN CONTENT FRAME ----
        main_frame = ctk.CTkFrame(self, corner_radius=10)
        main_frame.pack(side="top", fill="both", expand=True, pady=5, padx=5)

        # Left side - Canvas
        canvas_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.canvas = ctk.CTkCanvas(canvas_frame, bg="#181a1b", width=1000, height=700, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        self.canvas.bind("<Configure>", lambda e: self.update_canvas_image())

        # Right side - Transcription display
        transcription_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        transcription_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Transcription header
        transcription_header = ctk.CTkLabel(transcription_frame, text="Transcription Results", font=("Arial", 18, "bold"))
        transcription_header.pack(pady=10)

        # Transcription text area with scrollbar
        transcription_text_frame = ctk.CTkFrame(transcription_frame, corner_radius=10)
        transcription_text_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.transcription_text = tk.Text(transcription_text_frame, wrap=tk.WORD, bg="#23272e", fg="#e0e0e0", font=("Consolas", 13), relief=tk.FLAT, borderwidth=0, insertbackground="#e0e0e0")
        self.transcription_text.pack(side="left", fill="both", expand=True, padx=2, pady=2)

        transcription_scrollbar = ctk.CTkScrollbar(transcription_text_frame, command=self.transcription_text.yview)
        transcription_scrollbar.pack(side="right", fill="y")
        self.transcription_text.configure(yscrollcommand=transcription_scrollbar.set)

        self.transcription_text_frame = transcription_text_frame
        self.transcription_frame = transcription_frame
        self.transcription_header = transcription_header
        self.transcription_loading_bar = None

        # ---- BOTTOM FRAME ----
        bottom_frame = ctk.CTkFrame(self, corner_radius=10)
        bottom_frame.pack(side="bottom", fill="x", pady=5, padx=5)

        bottom_left_frame = ctk.CTkFrame(bottom_frame, corner_radius=10)
        bottom_left_frame.pack(side="left", padx=10, pady=2)

        self.prev_page_btn = ctk.CTkButton(bottom_left_frame, image=self.icon_prev, text="Prev (A)", compound="left", command=self.prev_page, width=120, font=("Arial", 12))
        self.prev_page_btn.pack(side="left", padx=5)
        self.next_page_btn = ctk.CTkButton(bottom_left_frame, image=self.icon_next, text="Next (S)", compound="left", command=self.next_page, width=120, font=("Arial", 12))
        self.next_page_btn.pack(side="left", padx=5)

        bottom_right_frame = ctk.CTkFrame(bottom_frame, corner_radius=10)
        bottom_right_frame.pack(side="right", padx=10, pady=2)

        self.export_only_btn = ctk.CTkButton(bottom_right_frame, image=self.icon_export, text="Export (E)", compound="left", command=self.export_segments_only, width=120, font=("Arial", 12))
        self.export_only_btn.pack(side="left", padx=5)
        self.export_page_btn = ctk.CTkButton(bottom_right_frame, image=self.icon_exportnext, text="Export & Next (R)", compound="left", command=self.export_and_next, width=160, font=("Arial", 12))
        self.export_page_btn.pack(side="right", padx=5)

        # Tooltips (simple implementation)
        self.add_tooltip(self.menu_new_btn, "Start a new project")
        self.add_tooltip(self.menu_open_btn, "Open an existing project")
        self.add_tooltip(self.menu_save_btn, "Save the current project")
        self.add_tooltip(self.menu_exit_btn, "Exit the application")
        self.add_tooltip(self.remove_last_segment_btn, "Remove the last segment")
        self.add_tooltip(self.split_last_segment_btn, "Split the last segment into sub-segments")
        self.add_tooltip(self.clear_segments_btn, "Clear all segments on this page")
        self.add_tooltip(self.switch_mode_btn, "Switch between Add and Edit mode")
        self.add_tooltip(self.toggle_input_mode_btn, "Toggle between Drag and Click input mode")
        self.add_tooltip(self.add_scan_page_btn, "Add a new scan page")
        self.add_tooltip(self.transcribe_btn, "Transcribe the current segments using AI")
        self.add_tooltip(self.export_only_btn, "Export segments as images")
        self.add_tooltip(self.export_page_btn, "Export segments and go to the next page")

        self.bind_keybindings()
        self.load_page_image()

    def add_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        label = tk.Label(tooltip, text=text, background="#222", foreground="#fff", relief="solid", borderwidth=1, font=("Arial", 10))
        label.pack(ipadx=4, ipady=2)
        def enter(event):
            x = widget.winfo_rootx() + 40
            y = widget.winfo_rooty() + 30
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()
        def leave(event):
            tooltip.withdraw()
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def copy_transcription(self):
        self.clipboard_clear()
        self.clipboard_append(self.transcription_text.get("1.0", tk.END))
        self.status_var.set("Transcription copied to clipboard!")

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

        # Add transcription shortcut
        self.bind_all('<t>', lambda e: self.transcribe_segments())

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

    def menu_new_project(self):
        if messagebox.askyesno("New Project", "Are you sure you want to start a new project? Unsaved changes will be lost."):
            def on_project_selected(pm):
                self.project_manager = pm
                self.init_main_ui()
            StartWindow(self, on_project_selected)

    def menu_open_project(self):
        if messagebox.askyesno("Open Project", "Are you sure you want to open a different project? Unsaved changes will be lost."):
            def on_project_selected(pm):
                self.project_manager = pm
                self.init_main_ui()
            StartWindow(self, on_project_selected)

    def menu_save_project(self):
        if self.project_manager:
            self.project_manager.save_project()
            messagebox.showinfo("Save Project", "Project saved successfully.")

    def transcribe_segments(self):
        """Transcribe the current page's segments and display the results."""
        if not self.current_scan_page_number:
            messagebox.showwarning("Warning", "No scan page selected.")
            return

        segments = self.segment_manager.get_segments_by_scan_page(self.current_scan_page_number)
        if not segments:
            messagebox.showwarning("Warning", "No segments to transcribe.")
            return

        temp_dir = os.path.join(self.project_manager.project_path, "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)

        # Show loading bar in place of text
        self.transcription_text.pack_forget()
        if self.transcription_loading_bar is None:
            self.transcription_loading_bar = ctk.CTkProgressBar(self.transcription_text_frame, width=300, height=24, corner_radius=12, mode="indeterminate")
        self.transcription_loading_bar.pack(expand=True, pady=40)
        self.transcription_loading_bar.start()
        self.transcription_frame.update()

        def do_transcription():
            try:
                segment_images = []
                for i, segment in enumerate(segments):
                    points = self.get_rotated_points(segment["original_points"])
                    img_path = os.path.join(temp_dir, f"segment_{i}.png")
                    self.export_segment(points, img_path)
                    segment_images.append(img_path)

                pdfname = os.path.splitext(os.path.basename(self.pdf_manager.get_current_pdf_path()))[0]
                current_page = self.pdf_manager.get_current_page_index() + 1
                messages = self.build_prompt_context(pdfname, current_page, segment_images)

                transcript = self.call_api(messages)

                transcript_dir = os.path.join(self.project_manager.project_path, "transcripts")
                os.makedirs(transcript_dir, exist_ok=True)
                transcript_file = os.path.join(transcript_dir, f"{pdfname}.txt")
                with open(transcript_file, "a", encoding="utf-8") as f:
                    f.write(f"PAGE {current_page}\n")
                    f.write(transcript.strip() + "\n\n")

                def on_success():
                    self.transcription_loading_bar.stop()
                    self.transcription_loading_bar.pack_forget()
                    self.transcription_text.pack(side="left", fill="both", expand=True, padx=2, pady=2)
                    self.transcription_text.delete(1.0, tk.END)
                    self.transcription_text.insert(tk.END, transcript)
                self.after(0, on_success)
            except Exception as e:
                def on_error():
                    if self.transcription_loading_bar:
                        self.transcription_loading_bar.stop()
                        self.transcription_loading_bar.pack_forget()
                    self.transcription_text.pack(side="left", fill="both", expand=True, padx=2, pady=2)
                    messagebox.showerror("Error", f"Transcription failed: {str(e)}")
                self.after(0, on_error)
            finally:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)

        threading.Thread(target=do_transcription, daemon=True).start()

    def build_prompt_context(self, pdfname: str, current_page: int, segment_images: List[str]) -> List[dict]:
        """Build the prompt context for the transcription API."""
        # Get previous transcripts
        transcript_dir = os.path.join(self.project_manager.project_path, "transcripts")
        transcript_file = os.path.join(transcript_dir, f"{pdfname}.txt")
        
        prev_transcripts = []
        if os.path.exists(transcript_file):
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
            
            if current_page_number is not None:
                page_transcripts[current_page_number] = "\n".join(current_page_lines)
            
            previous_pages = [p for p in page_transcripts.keys() if p < current_page]
            previous_pages.sort(reverse=True)
            last_3_pages = previous_pages[:3]
            
            last_3_pages.sort()
            for p in last_3_pages:
                prev_transcripts.append(f"Previous PAGE {p}:\n{page_transcripts[p]}")

        context_block = "\n\n".join(prev_transcripts) if prev_transcripts else "No previous context available."

        # Prepare image items
        image_items = []
        for img_path in segment_images:
            with open(img_path, "rb") as img_file:
                base64_str = base64.b64encode(img_file.read()).decode("utf-8")
            image_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_str}"
                }
            })

        # Build messages
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

    def call_api(self, messages: List[dict]) -> str:
        """Call the OpenAI API for transcription."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 1,
            "top_p": 1.0,
            "max_tokens": 2000
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")

        resp_json = response.json()
        choices = resp_json.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from API.")
        return choices[0]["message"]["content"].strip()
