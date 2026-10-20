# IIT Dharwad Academic Office Assistant

A local RAG app that indexes academic PDFs, answers questions through Groq, and cites source filename and PDF page number. It uses PyMuPDF for digital text, Tesseract OCR for image-based pages, MiniLM embeddings, and a persistent ChromaDB collection.

## Setup

1. Use Python 3.10 or newer and install Tesseract OCR separately. On Windows, install the Tesseract executable and set `TESSERACT_CMD` in `.env` if it is not on `PATH`.
2. Create an environment and install Python dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Create `documents/` beside these files and place the official PDF files there. If that folder is missing, ingestion automatically uses PDFs located beside `ingest.py` as a first-run convenience; existing PDFs are never moved.
4. Copy `.env.example` to `.env` and set `GROQ_API_KEY` to a valid Groq key.
5. Build/update the local vector index and start the UI:

   ```powershell
   python ingest.py
   streamlit run app.py
   ```

The first ingestion downloads the embedding model into the local Hugging Face cache. The vector database is persisted under `chroma_db/` by default. Stop the Streamlit app and rerun ingestion after adding, changing, or deleting source PDFs. Ingestion rebuilds the collection from the current folder so stale text is removed; its log reports the PDF, page, and chunk counts.

## Where PDF parsing and preprocessing happen

These steps are implemented in `ingest.py` and run when you execute `python ingest.py`:

| Step | Function | What it does |
| --- | --- | --- |
| File discovery | `load_documents()` | Finds PDFs in the configured documents folder. |
| PDF parsing and OCR | `extract_pdf_pages()` | Extracts digital text with PyMuPDF; OCRs pages with fewer than 50 extracted characters. Adds filename, page, path, category, and content-type metadata. |
| Table preprocessing | `_table_documents()` | Detects PDF tables and creates separate row-level records with repeated headers so semester/category labels stay with their values. |
| Chunking | `chunk_documents()` | Applies `RecursiveCharacterTextSplitter` with 700-character chunks and 120-character overlap. |
| Embedding and storage | `get_embeddings()` and `ingest()` | Embeds chunks locally with MiniLM and rebuilds the persistent Chroma collection. |

At query time, `rag_chain.py` classifies clear fee, schedule, rule, and program questions, searches the matching category, and combines semantic retrieval with exact-term matching. The assistant receives only the selected passages, not entire PDFs. The Streamlit page citations point back to those passages.

## Configuration

`DOCUMENTS_DIR`, `CHROMA_DIR`, `CHROMA_COLLECTION`, `GROQ_API_KEY`, `GROQ_MODEL`, `TOP_K`, `MAX_CONTEXT_CHARS`, `EMBEDDING_DEVICE`, `TESSERACT_CMD`, and `SIMILARITY_DISTANCE_THRESHOLD` can be set in `.env`. Retrieval returns up to 20 matching chunks by default across the full indexed collection. For fee-structure questions, the selected table's rows are added together so semester totals are not separated from their components. `MAX_CONTEXT_CHARS` caps the prompt context at 24,000 characters by default. The default Groq model is `openai/gpt-oss-120b`; check your Groq account's model access and billing before use. The threshold defaults to cosine distance `0.65` (lower is a closer match); tune it against the local documents and representative questions.

The chat's **Clear chat** button clears only the current conversation. It does not erase the index. Source accordions show the exact retrieved chunk, its filename, PDF page, and distance.
