"""
Comprehensive PDF parsing diagnostic.

Run this WHILE files are uploaded (or copy a test PDF to uploads/ first).
Usage:
  python debug_pdf.py                     # check uploads/ dir
  python debug_pdf.py path/to/test.pdf    # check a specific PDF
"""
import sys
import os
from pathlib import Path

def test_pdf(pdf_path: str):
    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"  ERROR: File not found: {pdf}")
        return
    print(f"\n{'='*60}")
    print(f"  FILE: {pdf.name}  ({pdf.stat().st_size:,} bytes)")
    print(f"{'='*60}")

    # Test 1: Raw pypdf extraction
    print("\n  [Test 1] pypdf direct extraction:")
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf))
        print(f"    Pages: {len(reader.pages)}")
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            print(f"    Page {i}: {len(text)} chars")
            if text.strip():
                print(f"    Preview: {repr(text[:300])}")
            else:
                print(f"    *** EMPTY PAGE — no text extracted ***")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")

    # Test 2: LlamaIndex SimpleDirectoryReader
    print("\n  [Test 2] LlamaIndex SimpleDirectoryReader:")
    try:
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[str(pdf)]).load_data()
        print(f"    Documents: {len(docs)}")
        for i, doc in enumerate(docs[:5]):
            print(f"    Doc[{i}]: {len(doc.text)} chars")
            if doc.text.strip():
                print(f"    Preview: {repr(doc.text[:300])}")
            else:
                print(f"    *** EMPTY DOC ***")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")

    # Test 3: Try PyMuPDF (better extraction for tricky PDFs)
    print("\n  [Test 3] PyMuPDF (fitz) extraction:")
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf))
        print(f"    Pages: {len(doc)}")
        for i, page in enumerate(doc):
            text = page.get_text()
            print(f"    Page {i}: {len(text)} chars")
            if text.strip():
                print(f"    Preview: {repr(text[:300])}")
            else:
                print(f"    *** EMPTY PAGE ***")
        doc.close()
    except ImportError:
        print(f"    PyMuPDF not installed (pip install PyMuPDF)")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")

    # Test 4: Check if PDF has images (could be scanned)
    print("\n  [Test 4] PDF structure analysis:")
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf))
        for i, page in enumerate(reader.pages[:3]):
            images = page.images if hasattr(page, 'images') else []
            resources = page.get("/Resources", {})
            has_font = "/Font" in resources if resources else False
            print(f"    Page {i}: fonts={has_font}, images={len(images)}")
            if not has_font and len(images) > 0:
                print(f"    *** LIKELY IMAGE-BASED PDF — needs OCR or PyMuPDF ***")
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────
print("PDF PARSING DIAGNOSTIC")
print(f"pypdf version: ", end="")
try:
    import pypdf
    print(pypdf.__version__)
except ImportError:
    print("NOT INSTALLED!")

# Check uploads dir
app_dir = Path(__file__).resolve().parent
upload_dir = app_dir / "uploads"
print(f"\nUploads directory: {upload_dir.resolve()}")
if upload_dir.exists():
    pdfs = list(upload_dir.rglob("*.pdf"))
    all_files = list(upload_dir.iterdir())
    print(f"  Top-level entries: {len(all_files)}")
    print(f"  PDF files: {len(pdfs)}")
    for f in all_files:
        if f.is_dir():
            print(f"    {f.name}/")
        else:
            print(f"    {f.name} ({f.stat().st_size:,} bytes)")
    for pdf in pdfs[:10]:
        print(f"    PDF: {pdf.relative_to(upload_dir)} ({pdf.stat().st_size:,} bytes)")
else:
    print("  Directory does not exist!")
    pdfs = []

# Check cache
cache_dir = app_dir / "cache"
if cache_dir.exists():
    print(f"\nCache directory: {len(list(cache_dir.iterdir()))} entries")
    pdf_caches = [d for d in cache_dir.iterdir() if ".pdf" in d.name]
    if pdf_caches:
        print(f"  PDF-related caches: {len(pdf_caches)}")
        for d in sorted(pdf_caches)[:6]:
            print(f"    {d.name}")

# If a specific file was given, test that
if len(sys.argv) > 1:
    test_pdf(sys.argv[1])
# Otherwise test all PDFs in uploads
elif pdfs:
    for pdf in pdfs:
        test_pdf(str(pdf))
else:
    print("\n*** No PDF files found in uploads/ ***")
    print("*** Either upload files via the web UI first, or run: ***")
    print("***   python debug_pdf.py path/to/your/test.pdf       ***")
