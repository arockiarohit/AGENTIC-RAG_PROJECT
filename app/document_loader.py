"""
Document loader module.
Handles parsing of PDF, DOCX, and TXT files into plain text.
"""

import os
import fitz  # PyMuPDF
from docx import Document
import json



def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        doc.close()
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {e}")
    return text


def load_docx(file_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {e}")
    return text


def load_txt(file_path: str) -> str:
    """Read plain text from a TXT file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
    except Exception as e:
        raise ValueError(f"Error reading TXT file: {e}")
    return text


def load_json(file_path: str) -> str:
    """Extract text from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert JSON to a pretty-printed string for processing
            return json.dumps(data, indent=2)
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {e}")


def load_document(file_path: str) -> str:
    """
    Load a document based on its file extension.
    Supports: .pdf, .docx, .txt
    Returns the extracted text content.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    elif ext == ".json":
        return load_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .docx, .txt, .json")
