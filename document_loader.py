"""
Document Loader — extracts plain text from various file formats.
"""

from pathlib import Path


def load_document(path: str) -> str:
    """Load a document and return its plain text content."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(p)
    elif suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".docx":
        return _load_docx(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(parts)


def _load_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    parts = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(parts)
