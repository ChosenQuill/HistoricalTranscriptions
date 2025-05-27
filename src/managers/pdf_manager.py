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
