"""Robust document loading with local-only PDF OCR.

SimpleDirectoryReader and pypdf only extract embedded PDF text. Scanned or
image-only PDFs have pixels but no text layer, so those readers correctly
return empty pages. This loader centralizes the fallback chain used by every
engine:

SimpleDirectoryReader -> PyMuPDF text -> pypdf text -> PyMuPDF OCR with Tesseract
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

def _total_chars(docs: Iterable[Document]) -> int:
    return sum(len(doc.text or "") for doc in docs)


def _get_config_value(name: str, default=None):
    try:
        from config import Config

        return getattr(Config, name, default)
    except Exception:
        return default


def _ocr_dpi() -> int:
    configured = os.getenv("PDF_OCR_DPI") or str(_get_config_value("PDF_OCR_DPI", 200))
    try:
        return max(120, min(int(configured), 300))
    except ValueError:
        return 200


def _tessdata_path() -> Path | None:
    configured = os.getenv("TESSDATA_PREFIX")
    candidates = [
        Path(configured) if configured else None,
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _page_documents(
    page_text: list[tuple[int, str]],
    file_path: Path,
    extraction_method: str,
) -> list[Document]:
    from llama_index.core.schema import Document

    docs = []
    for page_number, text in page_text:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        docs.append(
            Document(
                text=cleaned,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "page": page_number,
                    "extraction_method": extraction_method,
                },
            )
        )
    return docs


def _cache_table(
    rows: Iterable[tuple],
    file_path: Path,
    *,
    sheet_name: str,
    formula_rows: Iterable[tuple] | None = None,
) -> None:
    """Store one table exactly in SQLite; semantic documents are derived later."""
    rows = iter(rows)
    try:
        headers = next(rows)
    except StopIteration:
        return

    headers = [str(value).strip() if value is not None else f"Column {i + 1}" for i, value in enumerate(headers)]
    formula_iter = iter(formula_rows) if formula_rows is not None else None
    if formula_iter is not None:
        next(formula_iter, None)  # Skip the header row to stay aligned with values.

    materialized_rows = list(rows)
    materialized_formulas = list(formula_iter) if formula_iter is not None else None
    from spreadsheet_store import index_sheet

    index_sheet(file_path, sheet_name, headers, materialized_rows, materialized_formulas)


def _ingest_csv(file_path: Path) -> None:
    import csv

    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        _cache_table(csv.reader(handle), file_path, sheet_name="CSV")


def _ingest_xlsx(file_path: Path) -> None:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("XLSX support requires openpyxl. Install requirements.txt and restart the app.") from exc

    # Values answer financial questions; formulas expose model logic when requested.
    values_book = load_workbook(file_path, read_only=True, data_only=True)
    formulas_book = load_workbook(file_path, read_only=True, data_only=False)
    try:
        for values_sheet, formulas_sheet in zip(values_book.worksheets, formulas_book.worksheets):
            _cache_table(
                values_sheet.iter_rows(values_only=True),
                file_path,
                sheet_name=values_sheet.title,
                formula_rows=formulas_sheet.iter_rows(values_only=True),
            )
    finally:
        values_book.close()
        formulas_book.close()


def ingest_spreadsheet(file_path: str | Path, force: bool = False) -> int:
    """Parse a workbook once and make its exact rows immediately queryable."""
    file_path = Path(file_path)
    from spreadsheet_store import (
        begin_ingestion,
        cached_row_count,
        fail_ingestion,
        finish_ingestion,
        is_ready,
    )

    if not force and is_ready(file_path):
        return cached_row_count(file_path)
    begin_ingestion(file_path)
    try:
        if file_path.suffix.lower() == ".csv":
            _ingest_csv(file_path)
        elif file_path.suffix.lower() == ".xlsx":
            _ingest_xlsx(file_path)
        else:
            raise ValueError(f"Unsupported spreadsheet type: {file_path.suffix}")
        count = finish_ingestion(file_path)
        print(f"[spreadsheet_store] Ready: {file_path.name} ({count} rows)")
        return count
    except Exception as exc:
        fail_ingestion(file_path, exc)
        raise


def _load_spreadsheet_documents(file_path: Path) -> list[Document]:
    from llama_index.core.schema import Document
    from spreadsheet_store import semantic_records

    ingest_spreadsheet(file_path)
    return [Document(text=item["text"], metadata=item["metadata"]) for item in semantic_records(file_path)]


def _load_with_pymupdf_text(file_path: Path) -> list[Document]:
    import fitz

    pdf = fitz.open(str(file_path))
    try:
        pages = [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]
    finally:
        pdf.close()
    return _page_documents(pages, file_path, "pymupdf")


def _load_with_pypdf_text(file_path: Path) -> list[Document]:
    import pypdf

    reader = pypdf.PdfReader(str(file_path))
    pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return _page_documents(pages, file_path, "pypdf")


def _load_with_pymupdf_ocr(file_path: Path) -> list[Document]:
    """Use PyMuPDF's Tesseract bridge for local OCR."""
    import fitz

    tessdata = _tessdata_path()
    if not tessdata:
        raise RuntimeError(
            "Tesseract tessdata was not found. Install Tesseract OCR or set "
            "TESSDATA_PREFIX to the tessdata directory."
        )

    pdf = fitz.open(str(file_path))
    pages: list[tuple[int, str]] = []
    try:
        for i, page in enumerate(pdf):
            textpage = page.get_textpage_ocr(
                flags=0,
                language="eng",
                dpi=_ocr_dpi(),
                full=True,
                tessdata=str(tessdata),
            )
            pages.append((i + 1, page.get_text("text", textpage=textpage)))
    finally:
        pdf.close()
    return _page_documents(pages, file_path, "pymupdf_ocr")


def load_documents(file_path: str | Path) -> list[Document]:
    """Load a file into LlamaIndex Documents, OCRing scanned PDFs locally."""
    file_path = Path(file_path)
    fname = file_path.name
    is_pdf = fname.lower().endswith(".pdf")

    if file_path.suffix.lower() == ".csv":
        docs = _load_spreadsheet_documents(file_path)
        if not docs:
            raise ValueError(f"No rows were found in '{fname}'.")
        return docs
    if file_path.suffix.lower() == ".xlsx":
        docs = _load_spreadsheet_documents(file_path)
        if not docs:
            raise ValueError(f"No readable rows were found in '{fname}'.")
        return docs

    from llama_index.core import SimpleDirectoryReader

    docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
    total_chars = _total_chars(docs)
    print(f"[loader] Loaded '{fname}': {len(docs)} docs, {total_chars} chars")

    if total_chars == 0 and is_pdf:
        print(f"[loader] Empty PDF text for '{fname}', trying PyMuPDF text...")
        try:
            docs = _load_with_pymupdf_text(file_path)
            total_chars = _total_chars(docs)
            if total_chars:
                print(
                    f"[loader] PyMuPDF extracted '{fname}': "
                    f"{len(docs)} pages, {total_chars} chars"
                )
        except Exception as exc:
            print(f"[loader] PyMuPDF text failed for '{fname}': {exc}")

    if total_chars == 0 and is_pdf:
        print(f"[loader] Trying pypdf text extraction for '{fname}'...")
        try:
            docs = _load_with_pypdf_text(file_path)
            total_chars = _total_chars(docs)
            if total_chars:
                print(
                    f"[loader] pypdf extracted '{fname}': "
                    f"{len(docs)} pages, {total_chars} chars"
                )
        except Exception as exc:
            print(f"[loader] pypdf text failed for '{fname}': {exc}")

    if total_chars == 0 and is_pdf:
        print(f"[loader] PDF '{fname}' appears image-only, trying local OCR...")
        try:
            docs = _load_with_pymupdf_ocr(file_path)
            total_chars = _total_chars(docs)
            if total_chars:
                print(
                    f"[loader] Local OCR extracted '{fname}': "
                    f"{len(docs)} pages, {total_chars} chars"
                )
        except Exception as exc:
            raise RuntimeError(
                f"Could not extract text from '{fname}'. It looks like an image-only "
                "or scanned PDF, and local OCR failed. Install Tesseract OCR and "
                "make sure both tesseract.exe and tessdata are available. On "
                "Windows: winget install UB-Mannheim.TesseractOCR, then set "
                "TESSDATA_PREFIX to C:\\Program Files\\Tesseract-OCR\\tessdata. "
                f"Original OCR error: {exc}"
            ) from exc

    if is_pdf and total_chars == 0:
        raise ValueError(
            f"No readable text was found in '{fname}'. The PDF may be blank, "
            "encrypted, or a scan with unreadable images."
        )

    if docs and docs[0].text.strip():
        print(f"[loader] Preview for '{fname}': {repr(docs[0].text[:200])}")

    return docs
