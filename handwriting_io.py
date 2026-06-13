"""PDF → raster pages for handwriting OCR (PyMuPDF)."""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional


def pdf_bytes_to_temp_png_paths(
    pdf_bytes: bytes,
    zoom: float = 2.0,
    max_pages: int = 30,
) -> List[str]:
    """
    Render each PDF page to a temporary PNG path.
    Caller should delete paths when done (see cleanup_temp_paths).
    """
    if not pdf_bytes:
        return []
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PDF support requires PyMuPDF. Install: pip install pymupdf"
        ) from e

    out: List[str] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        n = min(int(doc.page_count), max(1, int(max_pages)))
        mat = fitz.Matrix(float(zoom), float(zoom))
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            fd, path = tempfile.mkstemp(suffix=f"_hw_p{i}.png")
            os.close(fd)
            pix.save(path)
            out.append(path)
    finally:
        doc.close()
    return out


def cleanup_temp_paths(paths: Optional[List[str]]) -> None:
    if not paths:
        return
    for p in paths:
        if p and os.path.isfile(p):
            try:
                os.unlink(p)
            except OSError:
                pass
