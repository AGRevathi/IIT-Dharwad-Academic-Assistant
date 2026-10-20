"""Ingest academic PDFs from ./documents into a persistent Chroma collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable

import pymupdf
import pytesseract
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from langchain_chroma import Chroma

load_dotenv()
if os.getenv("TESSERACT_CMD"):
    # pytesseract is a wrapper; the native Tesseract executable must be installed separately.
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("iitd_rag.ingest")

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "documents"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "iitd_academic_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
OCR_MIN_TEXT_CHARS = 50


def classify_document(filename: str) -> str:
    """Assign a broad routing label from the official document's filename."""
    name = filename.casefold()
    if any(term in name for term in ("fee", "tuition", "payment")):
        return "fees"
    if any(term in name for term in ("calendar", "timetable", "schedule", "mid semester", "holiday")):
        return "schedules"
    if any(term in name for term in ("rule", "ordinance", "regulation", "instruction")):
        return "rules"
    if any(term in name for term in ("syllabus", "course", "program", "programme", "index")):
        return "programs"
    return "general"


def _table_documents(page: pymupdf.Page, metadata: dict) -> list[Document]:
    """Convert detected PDF table rows into compact, self-describing text records."""
    try:
        tables = page.find_tables().tables
    except Exception as exc:
        LOGGER.debug("Table extraction unavailable on page %s: %s", metadata.get("page"), exc)
        return []

    output: list[Document] = []
    page_heading = " ".join(page.get_text("text").splitlines()[:8])
    title_match = re.search(r".{0,40}(?:fee structure|academic components|tuition).{0,60}",
                            page_heading, flags=re.IGNORECASE)
    table_title = title_match.group(0).strip() if title_match else "Academic document table"
    amount_pattern = re.compile(r"^(?:₹\s*)?(?:\d[\d,]*(?:\.\d+)?|[-–—])$")

    for table_number, table in enumerate(tables, start=1):
        raw_rows = table.extract()
        normalized_rows = [
            [" ".join(str(cell).split()) for cell in row if cell is not None and str(cell).strip()]
            for row in raw_rows
        ]
        normalized_rows = [row for row in normalized_rows if row]
        if len(normalized_rows) < 2:
            continue

        # The first few rows contain the column and category headings. Repeat them
        # in each indexed row record so chunk boundaries cannot separate values
        # from their semester/category labels.
        header_rows = normalized_rows[:min(4, len(normalized_rows))]
        header_text = "\n".join(" | ".join(row) for row in header_rows)
        records: list[dict[str, str]] = []
        section = "General"
        for row in normalized_rows[len(header_rows):]:
            value_index = next((i for i, cell in enumerate(row) if amount_pattern.fullmatch(cell)), None)
            if value_index is None:
                if len(row) >= 2 and re.fullmatch(r"[A-F]", row[0], flags=re.IGNORECASE):
                    section = " ".join(row)
                elif records:
                    # PyMuPDF sometimes puts the last word of a long row label on
                    # its own line, such as "Service Charges".
                    records[-1]["item"] += " " + " ".join(row)
                continue

            item = " ".join(row[:value_index]).strip(" |*\t") or "Unlabeled row"
            values = " | ".join(row[value_index:])
            records.append({"section": section, "item": item, "values": values})

        for row_number, record in enumerate(records, start=1):
            table_text = (
                f"Table title: {table_title}\n"
                f"Column headings from the source table, in left-to-right order:\n{header_text}\n"
                f"Section: {record['section']}\n"
                f"Item: {record['item']}\n"
                f"Values from left to right: {record['values']}"
            )
            row_metadata = {**metadata, "content_type": "table_row",
                            "table_number": table_number, "table_row": row_number}
            output.append(Document(page_content=table_text, metadata=row_metadata))
    return output


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


def extract_pdf_pages(pdf_path: Path) -> list[Document]:
    """Extract digital text page-by-page and OCR pages with little/no text."""
    pages: list[Document] = []
    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as exc:
        LOGGER.exception("Could not open %s", pdf_path)
        raise RuntimeError(f"Unable to open PDF: {pdf_path}") from exc

    with pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if len(text) < OCR_MIN_TEXT_CHARS:
                try:
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    text = pytesseract.image_to_string(image).strip()
                    LOGGER.info("OCR applied to %s page %d", pdf_path.name, page_index + 1)
                except pytesseract.TesseractNotFoundError as exc:
                    raise RuntimeError(
                        "Tesseract OCR executable was not found. Install Tesseract and, "
                        "if needed, set TESSERACT_CMD to its executable path."
                    ) from exc
                except Exception as exc:
                    LOGGER.exception("OCR failed for %s page %d", pdf_path.name, page_index + 1)
                    raise RuntimeError(f"OCR failed on {pdf_path.name}, page {page_index + 1}") from exc

            if text:
                metadata = {
                    "source": pdf_path.name,
                    "page": page_index + 1,
                    "source_path": str(pdf_path.resolve()),
                    "category": classify_document(pdf_path.name),
                    "content_type": "page_text",
                }
                pages.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )
                pages.extend(_table_documents(page, metadata))
            else:
                LOGGER.warning("No text could be extracted from %s page %d", pdf_path.name, page_index + 1)
    return pages


def load_documents(directory: Path) -> list[Document]:
    """Read every PDF recursively from the configured input directory."""
    if not directory.exists():
        # Convenient first-run fallback: this project folder may already contain PDFs.
        # Keep an explicitly configured non-default directory strict so typos are visible.
        if directory.resolve() == DOCUMENTS_DIR.resolve() == (BASE_DIR / "documents").resolve():
            root_pdfs = sorted(BASE_DIR.glob("*.pdf"))
            if root_pdfs:
                LOGGER.warning("%s is missing; ingesting PDFs found beside ingest.py. "
                               "Create documents/ to use that folder explicitly.", directory)
                directory = BASE_DIR
            else:
                raise FileNotFoundError(
                    f"Document directory does not exist: {directory}. Create it and add PDFs."
                )
        else:
            raise FileNotFoundError(f"Document directory does not exist: {directory}. Create it and add PDFs.")
    pdf_files = sorted(directory.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {directory}")
    documents: list[Document] = []
    for pdf_path in pdf_files:
        LOGGER.info("Extracting %s", pdf_path)
        documents.extend(extract_pdf_pages(pdf_path))
    if not documents:
        raise RuntimeError("No text was extracted from any PDF.")
    return documents


def chunk_documents(documents: Iterable[Document]) -> list[Document]:
    """Split each page independently, retaining source and page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def stable_chunk_id(document: Document, occurrence: int = 0) -> str:
    """Build deterministic IDs so re-ingesting the same inputs is idempotent."""
    identity = json.dumps(document.metadata, sort_keys=True, default=str)
    identity += f"|{document.page_content}|occurrence:{occurrence}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def ingest(directory: Path = DOCUMENTS_DIR) -> int:
    """Extract, chunk, embed, and persist all configured PDFs."""
    pages = load_documents(directory)
    chunks = chunk_documents(pages)
    if not chunks:
        raise RuntimeError("The PDFs did not produce any chunks.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    # This command indexes the complete current folder. Recreate the collection so
    # edits and deleted PDFs cannot leave stale chunks behind.
    vector_store.delete_collection()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    # A few PDFs contain identical extracted table rows with identical metadata.
    # Add a deterministic occurrence suffix so every Chroma ID remains unique.
    occurrences: dict[str, int] = {}
    chunk_ids = []
    for chunk in chunks:
        base_id = stable_chunk_id(chunk)
        occurrence = occurrences.get(base_id, 0)
        occurrences[base_id] = occurrence + 1
        chunk_ids.append(stable_chunk_id(chunk, occurrence))
    vector_store.add_documents(chunks, ids=chunk_ids)
    source_count = len({str(page.metadata.get("source_path", "")) for page in pages})
    LOGGER.info("Rebuilt index from %d PDFs and %d pages into %d chunks in %s",
                source_count, len(pages), len(chunks), CHROMA_DIR)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_DIR,
                        help="PDF input directory (default: ./documents)")
    args = parser.parse_args()
    print(f"Ingested {ingest(args.documents)} chunks.")
