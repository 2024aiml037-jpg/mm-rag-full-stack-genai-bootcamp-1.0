# Productionizing the Multimodal RAG Application — Project Plan

Submission for `Assignment-05-RAG-MMRAG/Multimodal_RAG_Production_Assignment.pdf`.
This plan takes the V1 prototype in this repository (Streamlit UI calling parsing →
ingestion → retrieval → multimodal generation directly) and designs its evolution into a
secure, scalable, observable, multi-tenant product.

---

## 1. Where V1 stands today (honest baseline)

| Concern | V1 implementation | Production gap |
| --- | --- | --- |
| UI | `ui/app.py` (~2.2k lines) imports the pipeline in-process; parsing, index restore, chat, inspection all live in Streamlit session state | No API boundary, no horizontal scaling, session state lost on refresh/restart |
| Parsing | `src/parsing.py` `ComplexPDFParser` — PyMuPDF text, Tesseract OCR, pdfplumber tables, embedded/page images written to `data/parsed_pdf_output/...` | Synchronous, blocks the request; artifacts on local disk, so a second replica cannot serve them |
| Ingestion | `src/ingestion.py` `MultimodalDocumentIngestion` — `RecursiveCharacterTextSplitter` (2000/120) for text, tables/images kept whole, `OpenAIEmbeddings`, upsert to one Qdrant collection with deterministic `uuid5` point ids | Dense-only vectors, no sparse index, no tenant fields, replace-whole-document only |
| Retrieval | `src/retriever.py` `MultimodalQdrantRetriever` — dense `similarity_search` with payload filters on `document_id`, `filename`, `file_sha256`, `content_type`, `page_number` | No sparse/BM25, no fusion, no reranker, no identity in the filter |
| Generation | `src/generation.py` `MultimodalRAGGenerator` — builds text/table/image context, inlines up to 4 images as base64 data URLs, prompts `gpt-4.1-mini` via `prompt_library/prompt.py` | Images are only findable through their OCR text; no eval harness; no per-request cost/latency accounting |
| Identity | none | Anyone with the app can read every indexed document |
| Ops | `logger/custom_logger.py` (structlog) + `exception/custom_exception.py` | Logs only; no traces, no metrics, no dashboards, no evals |

Two structural strengths of V1 are worth preserving because the whole plan builds on them:
deterministic `uuid5` point ids (idempotent re-ingestion) and a `content_type` discriminator
(`page_text_plus_ocr` / `table` / `image`) already present on every payload.

---

## 2. Target production architecture

```
                                   Users (web / mobile)
                                            |
                              Next.js frontend  (+ Streamlit kept as admin/demo)
                                            |
                            API Gateway  ->  Auth (OIDC/JWT, tenant claims)
                                            |
                 +--------------------------+---------------------------+
                 |                          |                           |
          Document API                  Chat API                  Admin/Analytics API
                 |                          |                           |
        presign + register          Retrieval Service            usage, audit, quotas
                 |                          |
          Object Storage (S3)       dense + sparse search  -> RRF -> Reranker
                 |                          |
        Job Queue (Redis/SQS)        Generation Service (multimodal LLM)
                 |                          |
        Parser Workers (Celery)      Answer + citations + trace_id
         |        |        |
      text     tables    images(VLM caption)
         |        |        |
        Embedding workers (text + sparse + optional image vectors)
                 |
              Qdrant  <-------- payload: tenant/workspace/doc/version/ACL
                 |
        PostgreSQL (users, tenants, workspaces, documents, versions, jobs,
                    permissions, chats, feedback, audit, usage)
                 |
        Observability: OpenTelemetry traces, Prometheus metrics, eval store
```

### 2.1 Component responsibilities

| Component | Responsibility | Scales on | Notes |
| --- | --- | --- | --- |
| Frontend (Next.js) | Upload, workspace/document browsing, chat, page preview, citations, feedback | Static/CDN | Streamlit `ui/app.py` demoted to internal admin/demo client of the same API |
| API Gateway / BFF (FastAPI) | AuthN, request validation, rate limits, quota checks, routing; no heavy CPU work | Requests/sec | Stateless; JWT carries `tenant_id`, `user_id`, roles |
| Document API | Presigned upload URLs, document + version registration, job enqueue, status polling | Requests/sec | Never parses inline |
| Ingestion/Parser workers | `ComplexPDFParser` refactored to read from and write to object storage; per-stage checkpointing | Queue depth, CPU (OCR-bound) | Separate queue for OCR-heavy work; dedicated pool avoids starving small docs |
| Embedding workers | Dense text embeddings, sparse (BM25/SPLADE) vectors, VLM image captions | Queue depth, model throughput | Batched, retryable, cost-metered |
| Object storage (S3/MinIO) | Original PDFs, page images, extracted images, parse manifests | Storage | Replaces `data/parsed_pdf_output/` local dependency |
| PostgreSQL | Users, tenants, workspaces, documents, versions, jobs, permissions, chats, messages, feedback, audit, usage | Read replicas | Source of truth for everything non-vector |
| Qdrant | Vector retrieval + filterable payload only | Shards/replicas per collection | Payload-based tenant partitioning, not a collection per user |
| Retrieval service | Filter construction from identity, dense + sparse query, RRF fusion, rerank, evidence selection | CPU/GPU for reranker | The ACL filter is built server-side from the token, never from the client body |
| Generation service | Prompt assembly, image attachment, multimodal LLM call, citation post-check, streaming | LLM concurrency | Isolated so LLM latency/failures don't block retrieval |
| Observability | Traces per request stage, quality/latency/cost metrics, eval runs, dashboards, alerts | — | Every answer carries a `trace_id` surfaced in the UI |

### 2.2 Request flows

**Ingestion (async).** `POST /documents` → presigned PUT → client uploads → `POST /documents/{id}/commit`
computes/records SHA-256 → job row `queued` → parser worker (`parsing`) → chunk/caption stage
(`chunking`) → embed + upsert (`indexing`) → `ready`. UI polls `GET /documents/{id}` or receives
a websocket/SSE status event.

**Query (sync, streaming).** `POST /chats/{id}/messages` → gateway validates JWT + quota →
retrieval service builds a mandatory ACL filter → dense + sparse search (top 50 each) → RRF →
cross-encoder rerank → top 6–8 evidence blocks (+ ≤4 images resolved to presigned URLs) →
generation service streams a grounded answer with citations → persist message, evidence ids,
tokens, cost, latency, `trace_id`.

---

## 3. Multi-user / multi-tenant data model

PostgreSQL (all tables carry `tenant_id`; row-level security enforced with a per-request
`SET LOCAL app.tenant_id`):

```
tenants(id, name, plan, created_at)
users(id, tenant_id, email, status, created_at)
memberships(user_id, tenant_id, role)                  -- owner | admin | member | viewer
workspaces(id, tenant_id, name, created_by)
workspace_members(workspace_id, user_id, role)
documents(id, tenant_id, workspace_id, filename, mime, current_version_id,
          latest_sha256, status, created_by, created_at)
document_versions(id, document_id, version_no, sha256, object_key, page_count,
                  parser_version, chunking_version, embedding_model, status,
                  indexed_at, superseded_at)
jobs(id, tenant_id, document_version_id, type, state, attempt, max_attempts,
     last_error, dedupe_key, started_at, finished_at)
permissions(id, tenant_id, subject_type, subject_id, resource_type, resource_id, action)
chats(id, tenant_id, workspace_id, user_id, title, share_token, visibility)
messages(id, chat_id, role, content, evidence jsonb, trace_id, tokens_in, tokens_out,
         cost_usd, latency_ms)
feedback(id, message_id, user_id, rating, reason, comment)
audit_logs(id, tenant_id, actor_id, action, resource, metadata jsonb, ip, created_at)
usage_counters(tenant_id, period, pages_parsed, tokens_in, tokens_out, queries, cost_usd)
```

Qdrant payload (single collection `mm_rag_chunks`, payload partitioning; one collection per
large enterprise tenant only when contractually required):

```json
{
  "tenant_id": "t_123", "workspace_id": "w_9", "document_id": "d_44",
  "document_version_id": "dv_7", "version_no": 3, "is_current": true,
  "acl_subjects": ["user:u_1", "role:member", "workspace:w_9"],
  "filename": "contracts.pdf", "page_number": 12,
  "content_type": "page_text_plus_ocr | table | image | image_caption | table_summary",
  "chunk_index": 4, "chunk_id": "...", "file_sha256": "...",
  "object_key": "tenant/t_123/doc/d_44/v3/page_012.png",
  "table_ref": "tbl_88", "language": "en", "parser_version": "p2", "embedding_model": "text-embedding-3-large"
}
```

Keyword indexes on `tenant_id`, `workspace_id`, `document_id`, `acl_subjects`, `content_type`,
`is_current`; integer index on `page_number`. Payload partitioning is chosen over
collection-per-user because Qdrant collections carry fixed HNSW/segment overhead — thousands of
collections waste memory and make cross-document retrieval impossible, while an indexed
`tenant_id` filter is applied inside the HNSW traversal at negligible cost.

**Why this differs from V1:** V1 indexes `metadata.document_id`/`filename`/`file_sha256` only.
Adding `tenant_id`, `workspace_id`, `acl_subjects`, `document_version_id`, and `is_current` is the
minimum payload change that makes both authorization and versioning expressible as filters.

---

## 4. Asynchronous ingestion workflow

```
uploaded -> queued -> parsing -> chunking -> embedding -> indexing -> ready
                 \                                              \
                  -> failed (terminal, with last_error)           -> partially_indexed (retryable)
                 \-> skipped_unchanged (hash match, no work done)
```

Per-stage design:

1. **uploaded** — client PUTs straight to object storage via presigned URL (API never streams file bytes); size/mime/page-count limits and virus scan hook.
2. **queued** — job row created with `dedupe_key = sha256 + parser_version + chunking_version + embedding_model`; a duplicate key returns the existing job instead of enqueuing work.
3. **parsing** — worker downloads the PDF to scratch, runs `ComplexPDFParser`, uploads `page_records.json`, `image_records.json`, `table_records.json`, page/extracted images, and a `parse_manifest.json` to `s3://.../doc/{id}/v{n}/`.
4. **chunking** — text split (current 2000/120 recursive splitter, now versioned as `chunking_version`), tables normalized (§6), images captioned by a VLM (§6).
5. **embedding** — batched dense + sparse vectors; per-batch retry; cost recorded to `usage_counters`.
6. **indexing** — idempotent upsert with deterministic `uuid5` ids (already in V1), then flip `is_current=true` for the new version and `false` for the previous one in a single filtered payload update.
7. **ready** — document becomes queryable; UI shows page count, table count, image count.

Reliability: at-least-once delivery with idempotent stages, exponential backoff (`1m, 5m, 25m`,
`max_attempts=3`), poison messages to a dead-letter queue with the full error chain from
`DocumentPortalException`, visibility timeout > p99 stage duration, heartbeats so a killed worker's
job is requeued, and a nightly reconciler that finds `indexing`-stuck versions whose Qdrant point
count ≠ expected and re-runs only that stage. Ingestion is never rolled back destructively: the old
version keeps serving traffic until the new version is fully indexed.

---

## 5. Retrieval design (dense → hybrid)

```
query
 ├─ ACL filter built server-side from JWT (tenant, workspace, acl_subjects, is_current)
 ├─ dense search    (text-embedding-3-large, top 50)      -- semantics, paraphrase, synonyms
 ├─ sparse search   (BM25 / SPLADE in Qdrant, top 50)     -- exact ids, clause numbers, SKUs, names
 ├─ RRF fusion      score = Σ 1/(k + rank_i), k=60        -- rank-based, no score normalization needed
 ├─ rerank          cross-encoder (bge-reranker-v2 / Cohere), 100 -> top 8
 ├─ evidence policy dedupe by (document, page); cap per document; keep ≥1 table and ≥1 image if scored
 └─ generation      top 6–8 blocks + ≤4 images, hard context budget (V1 already caps at 30k chars)
```

Where each stage runs and what it fixes:

- **Dense** (Qdrant, in-process with the retrieval service): recall for reworded questions; fails on rare literal tokens.
- **Sparse** (Qdrant named sparse vector on the same points — no second datastore): recovers exact-match queries like "clause 7.3" or invoice numbers that dense embeddings blur.
- **RRF** (retrieval service, CPU-cheap): merges two incomparable score scales robustly; no tuning of score weights per corpus.
- **Reranker** (separate GPU/CPU service, autoscaled independently): fixes precision — the single biggest measurable quality win, since generation quality is dominated by what sits in the top 5.
- **Metadata filters**: identity filters are mandatory; user-facing filters (`content_type`, page range, document subset) reuse `MultimodalQdrantRetriever.build_filter`, which already supports them.
- **Query understanding** (optional, phase 5+): multi-query expansion and HyDE for short questions, conversational query rewriting using chat history.

---

## 6. Multimodal retrieval strategy

### Images

*Current limitation:* `parsing.py` indexes an image's OCR text; `generation.py` then attaches the
local file. A diagram with little or no text is effectively unsearchable.

*Proposed:* during the chunking stage, send each page/extracted image to a VLM and store a
structured caption (what it shows, entities, axis labels, trends, visible numbers) as an
`image_caption` chunk that points back to the same `object_key`. Index the caption densely and
sparsely; retrieval then finds the image by meaning, and generation still attaches the real pixels.
Cache captions by image hash so re-ingestion is free.

*Future advanced path:* native visual retrieval with late-interaction multivector models
(ColPali/ColQwen-style) using Qdrant multivectors — embed page images directly with no OCR or
caption step, `MaxSim` scoring over patch embeddings. Roll it out as an A/B alternative retriever
behind the same interface, comparing quality against the caption pipeline on the eval set before
committing (it costs materially more storage and compute).

### Tables

*Current limitation:* a table becomes one Markdown blob; row/column-level factual questions
("what was the September penalty amount?") depend on the LLM re-parsing Markdown.

*Proposed:* store four representations per table, linked by `table_ref`:

1. **Raw** cell matrix (already produced as `raw_table`) in object storage — the audit source.
2. **Schema/header** record: column names, inferred types, units, row count, page — indexed as keywords.
3. **LLM-generated table summary** (`table_summary` chunk) describing what the table contains — this is what dense retrieval matches on.
4. **Normalized structured rows** in PostgreSQL (`table_rows` as JSONB, or Parquet in object storage) so exact questions can be answered by a deterministic lookup/aggregation tool instead of by reading Markdown.

Generation gets the summary plus the specific matching rows (row-level sub-chunks with
`row_index`), which keeps numbers, dates, and amounts faithful and citable to
`[file.pdf p.12 table 2 row 5]`.

---

## 7. Security model

- **Authentication:** OIDC/OAuth2 (Auth0/Cognito/Keycloak) issuing short-lived JWTs with `tenant_id`, `user_id`, roles, and workspace scopes; refresh tokens rotated; optional SSO/SAML and 2FA for enterprise tenants. Service-to-service calls use mTLS or signed internal tokens.
- **Authorization:** RBAC (`owner/admin/member/viewer`) plus resource ACLs in `permissions`. Enforced in three places: API handler (can this user touch this document?), PostgreSQL row-level security, and — critically — the retrieval filter. The retrieval service builds `must=[tenant_id=…, workspace_id ∈ allowed, acl_subjects ANY of subject list, is_current=true]` from the token; client-supplied filters may only *narrow* that, never widen it. A regression test asserts that a query with tenant B's token returns zero points from tenant A.
- **Tenant isolation:** payload partitioning by default; separate collection (or separate Qdrant cluster) as a premium option; per-tenant object storage key prefixes with IAM policy conditions; per-tenant encryption keys where required.
- **Secrets:** no `.env` in production — AWS Secrets Manager / Vault with rotation, injected at runtime; keys never logged (structlog processors redact `api_key`, `authorization`, `token`); presigned URLs expire in minutes.
- **Data protection:** TLS in transit, SSE-KMS at rest, PII redaction option before embedding, configurable retention and hard-delete (purge object storage + Qdrant points + Postgres rows on tenant deletion, with a documented DSR/GDPR path).
- **Auditability:** append-only `audit_logs` for login, upload, view, query, share, export, permission change, and delete, including `trace_id` so an audit entry links to the exact retrieval evidence.
- **Abuse/robustness:** per-tenant rate limits and quotas, upload size/page caps, prompt-injection mitigation (untrusted document content is delimited and never treated as instructions), and output checks that citations exist in the supplied evidence.

---

## 8. Document versioning and incremental ingestion

- `sha256` per uploaded file (V1 already indexes `file_sha256`); on commit, compare against `documents.latest_sha256`.
- **Unchanged file + unchanged pipeline versions** → `skipped_unchanged`, zero parse/embed cost.
- **Unchanged file but new `parser_version` / `chunking_version` / `embedding_model`** → re-index only, from the stored parse artifacts (no re-parse, no re-OCR).
- **Changed file** → new `document_versions` row, parse to a new prefix, and compute a per-page/per-chunk content hash diff against the previous version: re-embed only added/changed chunks, delete points for removed chunks, and reuse vectors for identical chunks. This turns a 200-page contract with a two-page amendment into a two-page job.
- **Atomic cutover:** index the new version alongside the old, then flip `is_current` in one filtered payload update; queries filter on `is_current=true`, so users never see a half-indexed document. Old versions stay queryable through an explicit "as of version" filter and are garbage-collected by retention policy.
- Chat citations pin `document_version_id`, so an old answer's sources remain reproducible.

---

## 9. Evaluation and observability plan

### Offline evaluation

A versioned dataset in the repo (`evals/goldens/*.jsonl`) with, per item: `question`,
`expected_source` (file + page + optional table/row), `expected_answer`, `modality`
(text/table/image), `difficulty`. Target ≥100 items covering the two sample PDFs, with a
deliberate slice of table-fact, image-only, multi-document, and unanswerable questions.

| Layer | Metric | Target (initial) |
| --- | --- | --- |
| Retrieval | Recall@10, Precision@5, MRR, nDCG@10, hit rate of expected page | Recall@10 ≥ 0.90 |
| Reranking | top-5 precision lift over fusion-only | ≥ +10 points |
| Answer | groundedness/faithfulness, correctness (LLM-judge + human spot check), citation correctness | groundedness ≥ 0.95, citation accuracy ≥ 0.95 |
| Refusal | correct abstention on unanswerable questions | ≥ 0.90 |
| Latency | p50/p95 retrieval, rerank, generation, end-to-end | p95 end-to-end ≤ 6 s streaming first token ≤ 1.5 s |
| Cost | tokens and USD per query, per ingested page | tracked per tenant, alert on 2× drift |
| Reliability | ingestion failure rate, retry rate, query 5xx rate | ingest failure < 1%, query 5xx < 0.5% |

Runs execute in CI on every change to prompts, chunking, retrieval, or models; results are stored
with the config hash so releases are directly comparable, and a regression beyond a threshold
blocks the merge. Ragas/DeepEval or an in-house judge harness is fine — the requirement is that
each run records `{git_sha, config_hash, dataset_version, metrics}`.

### Online observability

OpenTelemetry spans for `upload → parse → chunk → embed → index` and
`filter → dense → sparse → fuse → rerank → generate`, with `tenant_id`, `document_id`, and
`trace_id` on every span; Prometheus/Grafana dashboards for queue depth and age, stage durations,
Qdrant latency and point counts, token/cost burn, quota consumption, thumbs-down rate; structured
logs (already structlog) shipped centrally; Sentry for exceptions. Alerts on queue age, ingestion
failure spike, retrieval p95, LLM error rate, cost anomaly, and a drop in the daily canary eval.
Thumbs-down messages are auto-triaged into a review queue and promoted into the golden set —
that feedback loop is the main long-term quality engine.

---

## 10. Phased roadmap

Effort is expressed in focused work-sessions rather than calendar time; the ordering matters more
than the numbers.

| Phase | Scope | Exit criteria | Effort |
| --- | --- | --- | --- |
| 1 (done) | V1: parse, dense index, multimodal generate, Streamlit | Working demo | — |
| 2 | FastAPI backend extracted from `ui/app.py`; auth (OIDC), users/tenants/workspaces/documents in PostgreSQL; Streamlit becomes an API client; multi-document retrieval | UI has zero direct pipeline imports; login required; two users cannot see each other's docs at the API layer | 2–3 |
| 3 | Async ingestion: object storage, queue, Celery parser/embedding workers, job states, retries, DLQ, status UI | 100-page PDF uploads without blocking; killed worker recovers | 2–3 |
| 4 | Tenant/RBAC filtering pushed into retrieval; payload migration adding tenant/workspace/acl/version fields; isolation tests + audit logs | Cross-tenant retrieval test returns 0 hits; every query filtered server-side | 1–2 |
| 5 | Hybrid retrieval: sparse vectors, RRF, cross-encoder reranker, evidence selection policy | Recall@10 and top-5 precision improve measurably vs. phase 4 on the golden set | 2 |
| 6 | Multimodal upgrade: VLM image captions, four-way table representation + row-level lookup; ColPali-style retrieval evaluated as an option | Image-only and table-fact eval slices improve; row-level citations render in the UI | 2–3 |
| 7 | Evaluation + observability: golden dataset, CI eval gate, OTel tracing, dashboards, feedback loop | A prompt/model change is accepted or rejected on numbers, not opinion | 2 |
| 8 | Production API + Next.js frontend, streaming chat, page preview, shareable chats, conversation history, deployment (containers, autoscaling, IaC, blue/green) | Public product usable without Streamlit; documented SLOs | 3–4 |
| 9 | Enterprise: Drive/SharePoint/S3 connectors, quotas, billing/subscriptions, admin analytics, SSO/SCIM, governance and retention | Tenant can self-serve, is metered, and is auditable | 3–4 |

**Why this order.** Phases 2–4 are non-negotiable prerequisites: without a service boundary,
async processing, and identity-aware retrieval, every later feature is built on something that
would have to be rewritten — and the multi-tenant leak is the one defect that is unacceptable in
production. Phase 5 comes next because reranking is the cheapest large quality gain and needs no
new data model. Phase 7 sits before the public frontend deliberately: shipping a consumer UI
without evals and tracing means quality regressions become invisible. Phases 8–9 are product and
commercial surface, valuable only once the core is trustworthy.

**Key trade-offs taken explicitly.** Payload partitioning over collection-per-tenant (memory and
cross-document retrieval, with dedicated collections as a premium escape hatch); caption-based
image retrieval before ColPali (10× cheaper, evaluated head-to-head later); hosted APIs
(OpenAI embeddings/LLM) first with a provider-abstraction seam so self-hosted models can be
swapped in when cost or data-residency demands it; keeping Streamlit as an admin surface rather
than deleting it, so internal debugging survives the frontend rewrite.

---

## 11. Rubric mapping

| Rubric area | Marks | Covered in |
| --- | --- | --- |
| Architecture & service separation | 20 | §2, §2.1, §2.2 |
| Ingestion & storage design | 15 | §4, §8 |
| Security & multi-tenancy | 15 | §3, §7 |
| Retrieval quality | 15 | §5 |
| Multimodal design | 10 | §6 |
| Evaluation & observability | 15 | §9 |
| Roadmap & product thinking | 10 | §10 |
