# Multi-Newspaper AI RAG Backend

An AI-powered newspaper search and question-answering backend built with **FastAPI**, **PostgreSQL + pgvector**, **Celery**, and the **OpenAI API**.

Upload multiple English digital newspaper PDFs and ask questions about their content. Every question is answered by searching **all successfully processed newspapers together** — a single global vector search with HNSW-accelerated cosine similarity — and generating a grounded, citation-backed answer via an OpenAI-powered RAG pipeline.

---

## 1. Project Overview

- Built-in web UI served at `http://localhost:8000/` — drag-and-drop PDF upload, live processing status, and a citation-aware chat interface (`frontend/`).
- **Multilingual Support**: Upload digital and scanned **Hindi** (Devanagari Unicode), **English**, and **mixed Hindi-English** newspaper PDFs.
- **Tesseract OCR Integration**: Scanned, image-only, or epaper PDFs automatically trigger OCR (`eng+hin`) with high-resolution rendering, ensuring newspaper scans are accurately ingested without manual pre-processing.
- Automatic **Language Detection** (`en`, `hi`, `mixed`, `unknown`) stored in database records.
- Legacy Hindi font-encoded PDF detection (Kruti Dev, Chanakya, Walkman, etc.) — garbled non-Unicode PDFs are safely rejected with actionable error messages instead of corrupting data.
- PDFs are processed asynchronously: extracted page by page, cleaned, chunked page-wise (preserving Devanagari danda punctuation), embedded, and stored in PostgreSQL with pgvector.
- Cross-language RAG: Search across all Hindi & English newspapers in a single global vector search, with questions and answers seamlessly supported in Hindi and English.
- Answers include accurate citations: newspaper filename, PDF page number, chunk ID, and relevance distance.
- Follow-up questions are supported through conversation history contextualization, but retrieval always runs fresh.

## 2. Architecture Diagram

```
                       ┌──────────────────────────────────────────┐
                       │                 Client                   │
                       └──────────────────┬───────────────────────┘
                                          │ HTTP (REST, /api/v1)
                       ┌──────────────────▼───────────────────────┐
                       │          FastAPI  (Controllers)          │
                       │  document / chat / health controllers    │
                       └──────────────────┬───────────────────────┘
                                          │
                       ┌──────────────────▼───────────────────────┐
                       │               Services                   │
                       │ document · pdf · cleaning · chunking ·   │
                       │ embedding · vector search · rag · llm ·  │
                       │ chat                                     │
                       └───────┬─────────────────┬────────────────┘
                               │                 │
               ┌───────────────▼─────┐   ┌───────▼────────────────┐
               │    Repositories     │   │   Celery Worker        │
               │ document/chunk/chat │   │ (async PDF pipeline)   │
               └───────────────┬─────┘   └───────┬────────────────┘
                               │                 │
        ┌──────────────────────▼─────────────────▼──────┐   ┌──────────┐
        │        PostgreSQL 16 + pgvector               │   │  Redis   │
        │  documents · document_chunks (VECTOR 1536,    │   │ (broker/ │
        │  HNSW cosine) · chat_sessions · chat_messages │   │ backend) │
        └───────────────────────────────────────────────┘   └──────────┘
                               ▲
                        ┌──────┴───────┐
                        │  OpenAI API  │
                        │ embeddings + │
                        │ chat models  │
                        └──────────────┘
```

## 3. MVC Architecture Responsibilities

| Layer | Location | Responsibility |
|---|---|---|
| **Model** | `app/models/` | SQLAlchemy 2.x table definitions, relationships, pgvector columns |
| **Controller** | `app/controllers/` | FastAPI endpoints, HTTP handling, Pydantic validation; no business logic |
| **Service** | `app/services/` | Business logic: ingestion, extraction, cleaning, chunking, embeddings, retrieval, RAG, chat |
| **Repository** | `app/repositories/` | All database CRUD and the global vector similarity query |
| **Core** | `app/core/` | Config, DB/Redis/Celery connections, logging, exceptions, security helpers |
| **Worker** | `app/workers/` | Celery background tasks: async PDF processing, retries, status updates |

## 4. Folder Structure

```
newspaper-rag-backend/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── core/                    # config, database, redis, celery, logging, exceptions, security
│   ├── models/                  # document, document_chunk, chat_session, chat_message
│   ├── schemas/                 # Pydantic v2 request/response schemas
│   ├── repositories/            # base, document, chunk (vector search), chat
│   ├── controllers/             # document, chat, health endpoints
│   ├── services/                # document, pdf, ocr_service, language_service, text_cleaning,
│   │                            # chunking, embedding, vector_search, rag, llm, chat
│   ├── workers/                 # document_tasks.py (Celery)
│   ├── api/v1/router.py         # versioned router
│   └── utils/                   # file, hash, response helpers
├── frontend/                    # single-page web UI (index.html, style.css, app.js)
├── alembic/                     # migrations (pgvector + HNSW in initial revision)
├── storage/newspapers/          # uploaded PDFs (Docker volume)
├── tests/                       # pytest suite (OpenAI mocked)
├── scripts/init_db.py           # dev convenience bootstrap
├── Dockerfile / docker-compose.yml
├── .env.example / alembic.ini / requirements.txt
```

## 5. Technology Stack

- **Python 3.12+**, **FastAPI**, **Pydantic v2**, **Uvicorn**
- **PostgreSQL 16** with the **pgvector** extension (`pgvector/pgvector:pg16` image)
- **SQLAlchemy 2.x** (async for API, sync for workers), **Alembic**
- **OpenAI API** — `text-embedding-3-small` (1536 dims) + `gpt-4.1-mini` (configurable)
- **LangChain Text Splitters** (`RecursiveCharacterTextSplitter`)
- **PyMuPDF & Tesseract OCR** (`pytesseract`, `tesseract-ocr-eng`, `tesseract-ocr-hin`) for digital and scanned PDF text extraction
- **Celery + Redis** for background processing
- **Pytest + HTTPX** for testing
- **Docker + Docker Compose**

## 6. PostgreSQL + pgvector

pgvector is an **extension inside PostgreSQL**, not a separate database. Relational data (documents, chat) and 1536-dimensional embeddings live in the **same database**, enabling a single SQL query to join chunks with document metadata and order by vector distance.

Embeddings are stored as native `VECTOR(1536)` columns — never JSON or text.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 7. HNSW + Cosine Distance

These are two different things used **together**:

- **HNSW** (Hierarchical Navigable Small World) is the *approximate nearest-neighbor index structure* — it makes similarity search fast.
- **Cosine distance** (`<=>` operator, `vector_cosine_ops` operator class) is the *distance metric* the index is built for.

```sql
CREATE INDEX idx_document_chunks_embedding
ON document_chunks
USING hnsw (embedding vector_cosine_ops);
```

Recall is tuned per-transaction with `SET LOCAL hnsw.ef_search = <n>` (configurable via `HNSW_EF_SEARCH`). Note: PostgreSQL may legitimately choose a sequential scan for very small datasets — that is expected planner behavior, not a bug. Verify index usage on real data with `EXPLAIN ANALYZE`.

## 8. Database Schema

| Table | Purpose | Key columns |
|---|---|---|
| `documents` | Uploaded newspapers | `id` UUID PK, `filename`, `stored_filename` (unique), `file_hash` (SHA-256, unique), `file_size`, `total_pages`, `total_chunks`, `status` (PROCESSING/COMPLETED/FAILED), `language` (`en`/`hi`/`mixed`/`unknown`), `error_message`, `embedding_model`, timestamps |
| `document_chunks` | Text + embeddings | `id` UUID PK, `document_id` FK (CASCADE), `chunk_text`, `embedding VECTOR(1536)`, `language` (`en`/`hi`/`mixed`), `page_number`, `chunk_index`; HNSW index + `document_id` index |
| `chat_sessions` | Conversations | `id` UUID PK, timestamps |
| `chat_messages` | Messages | `id` UUID PK, `session_id` FK (CASCADE), `role` (USER/ASSISTANT), `content`, `sources` JSONB, `created_at`; indexed on `session_id` and `created_at` |

## 9. PDF Ingestion Pipeline

```
Upload → validate (extension, MIME, %PDF magic, size) → SHA-256 duplicate check
      → store with unique safe filename → create PROCESSING record → Celery task
      → [worker] extract pages (PyMuPDF) → reject scanned/empty (no OCR)
      → clean text → page-wise chunking → batched embeddings → insert chunks
      → mark COMPLETED (or FAILED with error message)
```

- The upload API returns immediately (`202 Accepted`); heavy work happens in the worker.
- Scanned or image-only PDFs are rejected with a clear error (`MIN_EXTRACTABLE_CHARS` threshold).
- Processing is **idempotent**: previous chunks for the document are deleted before new ones are inserted, so retries and reprocessing never duplicate data, and failures never leave partial chunks behind.
- Transient errors (OpenAI rate limits, network) retry with exponential backoff (max 3 retries); content errors fail immediately.

## 10. Embedding Pipeline

- Model: `text-embedding-3-small`, 1536 dimensions (both configurable).
- Chunks embedded in batches (`EMBEDDING_BATCH_SIZE`, default 64) to reduce API overhead.
- Retry with exponential backoff (tenacity) for rate limits/timeouts.
- Vector dimensions validated before any database insert.
- The embedding model name is stored on the document for future re-embedding.
- Completed documents are never re-embedded unless explicitly reprocessed.

## 11. Global Newspaper Retrieval

**Every chat question searches all completed newspapers together.** There is no newspaper selector and no `document_id` filter in the retrieval query:

```sql
SELECT dc.id, dc.chunk_text, dc.page_number, dc.chunk_index,
       d.id AS document_id, d.filename,
       dc.embedding <=> :query_embedding AS distance
FROM document_chunks dc
JOIN documents d ON d.id = dc.document_id
WHERE d.status = 'COMPLETED'
ORDER BY dc.embedding <=> :query_embedding
LIMIT :top_k;
```

Implemented in `app/repositories/chunk_repository.py` via SQLAlchemy's `cosine_distance()` with a parameterized query. `top_k` defaults to 5 (`RETRIEVAL_TOP_K`). An optional cosine-distance cutoff (`RETRIEVAL_MAX_DISTANCE`) is disabled by default — calibrate it against real data before enabling (relevant matches with `text-embedding-3-small` typically fall well below 0.6 cosine distance).

## 12. RAG Chat Pipeline

```
Question → Chat Controller → Chat Service → load recent history
        → contextualize follow-up (LLM rewrite to standalone question)
        → embed query (OpenAI) → GLOBAL pgvector search (HNSW + cosine)
        → build numbered context from chunks → OpenAI chat completion
        → answer + citations → persist both messages → response
```

- Follow-up questions ("What are **its** effects on employment?") are rewritten into standalone questions using history, then **still searched globally** — history never replaces retrieval.
- The system prompt forces the LLM to: answer only from excerpts, never invent facts, admit insufficiency, treat newspaper content as untrusted data (prompt-injection resistant), and cite `[Source N]` inline.
- If no relevant chunks are found, the API honestly reports that the uploaded newspapers don't contain the information.

## 13. Citation Mechanism

Every answer includes `sources` built **from the actual retrieved chunks** (never invented):

```json
{
  "document_id": "uuid",
  "filename": "times_of_india.pdf",
  "page_number": 5,
  "chunk_id": "uuid",
  "relevance_distance": 0.21,
  "excerpt": "Relevant extracted newspaper content..."
}
```

Page numbers are accurate because chunking is **page-wise** — a chunk never mixes text from multiple PDF pages.

## 14. Docker Setup

Four services: `api`, `worker`, `db` (`pgvector/pgvector:pg16`), `redis`. Persistent volumes for PostgreSQL data, uploaded newspapers (shared between api and worker), and Redis. Health checks and dependency conditions ensure correct startup order.

```bash
# 1. Configure environment
cp .env.example .env
# edit .env: set POSTGRES_PASSWORD, OPENAI_API_KEY, matching DATABASE_URL

# 2. Build and start everything
docker compose up -d --build

# 3. Run database migrations
docker compose exec api alembic upgrade head

# 4. Check health
curl http://localhost:8000/api/v1/health

# 5. Open the web UI
# http://localhost:8000/
```

Common operations:

```bash
docker compose build                         # build services
docker compose up -d                         # start services
docker compose down                          # stop services
docker compose down -v                       # stop + remove volumes (destructive)
docker compose logs -f api                   # API logs
docker compose logs -f worker                # worker logs
docker compose exec api bash                 # shell into API container
docker compose exec db psql -U postgres -d newspaper_rag   # PostgreSQL shell
docker compose restart worker                # restart the worker
docker compose ps                            # service status + health
docker compose exec api alembic upgrade head # run migrations
```

## 15. Environment Configuration

All settings load through Pydantic Settings from `.env` (see [.env.example](./.env.example)). Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | — | asyncpg URL; sync URL is derived automatically |
| `OPENAI_API_KEY` | — | **required**; never logged or exposed |
| `OPENAI_EMBEDDING_MODEL` / `OPENAI_EMBEDDING_DIMENSIONS` | `text-embedding-3-small` / `1536` | embedding config |
| `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | answer-generation LLM |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | chunking |
| `RETRIEVAL_TOP_K` | `5` | chunks retrieved per question |
| `HNSW_EF_SEARCH` | `100` | HNSW recall tuning (transaction-local) |
| `RETRIEVAL_MAX_DISTANCE` | `0` (disabled) | optional distance cutoff |
| `MAX_UPLOAD_SIZE_MB` | `50` | upload limit |
| `MIN_EXTRACTABLE_CHARS` | `200` | scanned-PDF rejection threshold |
| `CORS_ORIGINS` | `*` | comma-separated allowed origins |

Never commit `.env` to Git (already ignored).

## 16. Alembic Migrations

Alembic reads the database URL from application settings (`alembic/env.py`), imports all models, and the initial migration creates the `vector` extension **before** the `VECTOR(1536)` column and HNSW index.

```bash
# Apply all migrations
docker compose exec api alembic upgrade head

# Roll back one revision
docker compose exec api alembic downgrade -1

# Create a new autogenerated revision
docker compose exec api alembic revision --autogenerate -m "describe change"

# Show current revision
docker compose exec api alembic current
```

## 17. API Endpoints

Base prefix: `/api/v1`. Interactive docs at `http://localhost:8000/docs` (when `DEBUG=true`).

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents/upload` | Multipart PDF upload → `202` with `document_id`, status `PROCESSING` |
| `GET` | `/documents?page=1&page_size=10` | Paginated document list with processing status |
| `GET` | `/documents/{document_id}` | Document details: status, pages, chunks, error info |
| `DELETE` | `/documents/{document_id}` | Delete record, chunks/embeddings (cascade), and the PDF file |
| `POST` | `/documents/{document_id}/reprocess` | Re-queue a failed/completed document (blocked while PROCESSING) |
| `POST` | `/chat` | Ask a question; `session_id: null` creates a new session |
| `GET` | `/chat/history/{session_id}` | Chronological conversation history |
| `GET` | `/health` | App + database + Redis health |

Example chat request/response:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": null, "question": "What are the latest developments in artificial intelligence?"}'
```

```json
{
  "session_id": "e8f1...",
  "question": "What are the latest developments in artificial intelligence?",
  "answer": "The newspapers report that ... [Source 1][Source 2]",
  "sources": [
    {
      "document_id": "a2c4...",
      "filename": "newspaper_a.pdf",
      "page_number": 4,
      "chunk_id": "b7d9...",
      "relevance_distance": 0.19,
      "excerpt": "Relevant newspaper content..."
    }
  ]
}
```

Error responses are consistent JSON:

```json
{ "success": false, "error": { "code": "DUPLICATE_DOCUMENT", "message": "..." } }
```

## 18. Running the Celery Worker

In Docker the worker starts automatically. To run it manually (local development):

```bash
celery -A app.core.celery_app:celery_app worker --loglevel=INFO --concurrency=2 -Q documents
```

Check worker status:

```bash
docker compose exec worker celery -A app.core.celery_app:celery_app inspect active
docker compose exec worker celery -A app.core.celery_app:celery_app inspect stats
```

## 19. Running Tests

Tests mock all OpenAI calls and use dependency overrides instead of a live database — no services required.

```bash
# local virtualenv
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
pytest tests -v

# or inside Docker
docker compose exec api pytest tests -v
```

The suite covers: health checks, upload validation (type/size/duplicate/fake PDF), digital PDF extraction, scanned/empty PDF rejection, text cleaning, page-wise chunking, mocked embedding generation + dimension validation, **global retrieval with explicit verification that no document filter is applied**, chat endpoint + sessions + history, citation metadata traceability, and document deletion with safe file cleanup.

## 20. Common Errors and Troubleshooting

| Problem | Cause / Fix |
|---|---|
| `SCANNED_OR_EMPTY_PDF` on upload processing | The PDF has no extractable text. Only digital/searchable PDFs are supported; OCR is intentionally not implemented. |
| `DUPLICATE_DOCUMENT` (409) | The same file (identical SHA-256) was already uploaded. Delete the existing document first if you need to re-ingest. |
| Document stuck in `PROCESSING` | Check worker logs: `docker compose logs -f worker`. Ensure `OPENAI_API_KEY` is set and Redis is healthy. Use the reprocess endpoint after fixing. |
| `OPENAI_SERVICE_ERROR` (502) | OpenAI rate limit or outage. The system retries with backoff; retry the request later. |
| `relation "document_chunks" does not exist` | Migrations not applied: `docker compose exec api alembic upgrade head`. |
| `type "vector" does not exist` | pgvector extension missing — make sure you use the `pgvector/pgvector:pg16` image; the initial migration creates the extension. |
| HNSW index not used (seq scan in `EXPLAIN`) | Normal for small datasets; PostgreSQL's planner prefers a sequential scan when the table is tiny. It switches to the index as data grows. |
| `POSTGRES_PASSWORD must be set` on `docker compose up` | Copy `.env.example` to `.env` and set a real password (also inside `DATABASE_URL`). |
| Worker can't find uploaded file | API and worker must share the `newspaper_storage` volume (the provided compose file does this). Don't change `UPLOAD_DIR` in one service only. |
| CORS errors in a browser client | Set `CORS_ORIGINS=https://your-frontend.example.com` in `.env`. |

## Security Notes

Implemented: file type/size/magic-byte validation, SHA-256 dedup, sanitized + UUID stored filenames, upload-path containment, parameterized queries, environment-based secrets, transaction management, OpenAI timeout/retry, prompt-injection hardening in the system prompt, internal errors never exposed to clients, secrets excluded from logs.

**Not yet implemented** (extension points are in `app/core/security.py`): authentication, authorization, rate limiting. Add these before any public deployment.



🌐 Access URLs (Browser में खोलें):
Service	URL	विवरण
Web Frontend	http://localhost:8000/	Newspaper PDF upload और Chat interface
Swagger API Docs	http://localhost:8000/docs	Interactive API testing documentation
Health Check	http://localhost:8000/api/v1/health	API, DB और Redis की health स्थिति
Database UI (Adminer)	http://localhost:8080/	System: PostgreSQL, Server: db, Username: postgres, DB: newspaper_rag