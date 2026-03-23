# Open Memory Benchmark (OMB)

Living documentation for the `memory-bench` project. Keep this file up to date as datasets, providers, and commands evolve.

## Project Overview

A benchmarking framework for evaluating memory/retrieval providers on long-context datasets (personal conversations, agent trajectories). The CLI is `amb` (invoked via `uv run amb`).

**Core concepts:**
- **Dataset** — source of documents and queries (e.g. `personamem`)
- **Split** — subset of a dataset (e.g. `32k`, `128k`)
- **Category** — optional sub-filter within a split (e.g. `question_type` in personamem)
- **Memory provider** — the system being benchmarked (e.g. `mem0-cloud`, `bm25`)
- **Mode** — how the provider is used to answer queries (default: `rag`)

**Output path:** `outputs/{dataset}/{memory}/{mode}/{split}.json` (or `{split}/{category}.json`)

---

## Datasets

### `personamem`
**HuggingFace:** `bowen-upenn/PersonaMem` · **Task:** MCQ (exact letter match)
**Splits:** `32k`, `128k`, `1M` (context-window sizes)

Long-horizon personal preference tracking. Conversation histories are split into sessions at each system turn (persona/time-period frame). Questions ask to pick the best assistant reply from 4 options.

- Documents: one per session, ID = `{shared_context_id}_{session_index}`
- Gold IDs: sessions whose start turn index < `end_index_in_shared_context`
- Categories: distinct `question_type` values (e.g. `recall user shared facts`, `infer user preferences`)

### `memsim`
**Source:** `nuster1128/MemSim` (GitHub raw download) · **Task:** MCQ
**Splits (= QA types):** `simple`, `conditional`, `comparative`, `aggregative`, `post_processing`, `noisy`

Chinese-language daily-life memory simulation. Each trajectory has a `message_list` (documents) and a single QA. Gold IDs map to `target_step_id` evidence messages.

### `tempo`
**HuggingFace:** `tempo26/Tempo` · **Task:** Open-ended (LLM judge)
**Splits:** `bitcoin`, `cardano`, `iota`, `monero`, `economics`, `law`, `politics`, `history`, `quant`, `travel`, `workplace`, `genealogy`, `hsm`

Time-sensitive QA. Tests whether a memory system correctly handles *when* information was recorded.

### `membench`
**Source:** Manual download from Google Drive · **Task:** Open-ended (LLM judge)
**Splits:** `FirstAgentLowLevel`, `FirstAgentHighLevel`, `ThirdAgentLowLevel`, `ThirdAgentHighLevel`

Agent memory benchmark at different abstraction levels and agent perspectives (first-person vs third-person). Data must be placed locally before running.

### `locomo`
**Source:** `snap-research/locomo` (GitHub raw download) · **Task:** Open-ended (LLM judge)
**Splits:** `locomo10` (single split, 10 conversations)
**Categories (query-level):** `single-hop`, `temporal`, `multi-hop`, `open-domain`, `adversarial`

Long-term multi-session conversation memory benchmark. 10 conversations spanning up to 9 sessions each with timestamps. 1,986 QA pairs total. Data is auto-downloaded from GitHub; set `LOCOMO_DATA_PATH` to use a local copy.

- Documents: one per session, ID = `{sample_id}_{session_key}` (e.g. `abc_session_1`)
- Gold IDs: sessions containing evidence turns (mapped via `dia_id` field)
- Categories are query-level only (integer 1–5 in source, mapped to names above)
- 272 session docs · ~282 single-hop, 321 temporal, 96 multi-hop, 841 open-domain, 446 adversarial

---

## Memory Providers

| Name | Type | Notes |
|------|------|-------|
| `bm25` | Local / keyword search | Baseline, no API needed |
| `qdrant` | Local / vector search | Local Qdrant instance, no API needed |
| `mem0` | Local / agentic memory | Uses Gemini LLM + HuggingFace embeddings + local Qdrant; requires `GOOGLE_API_KEY`. Set `on_disk=True` for persistent store. |
| `mem0-cloud` | Cloud / agentic memory | Uses mem0 API; requires `MEM0_CLOUD_KEY`. Async ingestion — waits for indexing before eval. k=20. |
| `hindsight` | Local / agentic memory | Embedded Hindsight with Gemini; requires `GOOGLE_API_KEY` |
| `hindsight-cloud` | Cloud / agentic memory | Hindsight cloud API; requires `HINDSIGHT_CLOUD_KEY` |
| `mastra` | Local / agentic memory | Mastra memory via subprocess |
| `mastra-om` | Local / agentic memory | Mastra OpenMemory variant |
| `cognee` | Local / graph memory | Cognee knowledge graph |
| `supermemory` | Cloud | Supermemory API |

**Key env vars:**
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` — used by Gemini-backed providers and the LLM judge
- `MEM0_CLOUD_KEY` — mem0 cloud API key
- `HINDSIGHT_CLOUD_KEY` — Hindsight cloud API key

---

## CLI Reference

```bash
# Run benchmark
uv run amb run --dataset personamem --split 32k --memory mem0-cloud

# With options
uv run amb run --dataset personamem --split 32k --memory mem0-cloud \
  --category "recall user shared facts" \
  --query-limit 20 \
  --oracle          # only ingest gold documents \
  --skip-ingestion  # skip ingest, reuse existing store

# List splits and categories
uv run amb splits --dataset personamem

# Dataset stats
uv run amb dataset-stats --dataset personamem

# View results in browser
uv run amb view
```

---

---

## Publishing Workflow

Results and dataset files are stored in git (source of truth) and mirrored to Vercel Blob (CDN for the public deployment). The server always checks local disk first, then falls back to Blob transparently.

### After a new benchmark run

```bash
uv run amb publish-results --push   # compress + upload outputs/ to Blob
git add outputs/ && git commit -m "results: <description>" && git push
```

### After adding or updating a dataset

```bash
uv run amb publish-dataset --dataset <name> --push   # export data/ + upload to Blob
git add data/ && git commit -m "data: export <name>" && git push
```

Both commands are **idempotent** (checksum-based) — safe to re-run, only uploads what changed.
`BLOB_READ_WRITE_TOKEN` must be set in `.env` (already configured).

**Without `--push`**: only compresses/exports locally, no Blob upload.
**`--force`**: re-upload everything regardless of checksums.

### Vercel deployment

- Deployed at: https://open-memory-benchmark-vectorize.vercel.app
- Static UI (`ui/dist/`) is committed to git and served directly
- `outputs/` is committed to git AND mirrored to Blob (Vercel reads from Blob when local files aren't in the Lambda bundle)
- `data/` is excluded from the Lambda bundle (too large), served entirely from Blob
- If Blob is lost: re-run `publish-results --push --force` and `publish-dataset --push --force` for each dataset

---

## Adding a New Provider

1. Create `src/memory_bench/memory/<name>.py` implementing `MemoryProvider` (`ingest`, `retrieve`)
2. Register it in `src/memory_bench/memory/__init__.py` → `REGISTRY`
3. Use `prepare(store_dir)` for any persistent local state (called before ingest/retrieve)

## Adding a New Dataset

1. Create `src/memory_bench/dataset/<name>.py` implementing `Dataset` (`load_queries`, `load_documents`)
2. Register it in `src/memory_bench/dataset/__init__.py`
3. Implement `categories(split)` if the dataset has sub-filters
