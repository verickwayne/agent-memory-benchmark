# AMB — Agent Memory Benchmark

A modular, open benchmark for evaluating memory systems on personal and long-horizon QA tasks.
Live leaderboard: **[agentmemorybenchmark.ai](https://agentmemorybenchmark.ai)**

## How it works

1. **Ingest** — documents from a dataset are loaded into a memory provider
2. **Retrieve** — for each query the memory provider retrieves relevant context
3. **Generate** — an LLM produces an answer from the retrieved context
4. **Judge** — a second LLM call scores the answer against gold answers

## Setup

```bash
git clone https://github.com/vectorize-io/agent-memory-benchmark
cd agent-memory-benchmark
cp .env.example .env   # add your API keys
uv sync
```

Required env vars (`.env`):
```
GEMINI_API_KEY=...          # used by the judge LLM and Gemini-backed providers
```

Optional:
```
AMB_ANSWER_LLM=gemini       # provider for answer generation: gemini | groq | openai (default: groq)
AMB_ANSWER_MODEL=...        # model override, e.g. gemini-2.5-pro-preview
AMB_JUDGE_LLM=gemini        # provider for judging (default: gemini)
AMB_JUDGE_MODEL=...         # judge model override
```

## Running a benchmark

```bash
# Run on a dataset + split
uv run amb run --dataset personamem --split 32k --memory hindsight

# Limit queries for a quick smoke-test
uv run amb run --dataset personamem --split 32k --memory bm25 --query-limit 20

# Resume a previous run (skip already-evaluated queries)
uv run amb run --dataset personamem --split 32k --memory hindsight --skip-ingested

# Browse results locally
uv run amb view
```

## Datasets

| Dataset | Splits | Task |
|---|---|---|
| `personamem` | `32k`, `128k`, `1M` | Long-horizon preference tracking MCQ |
| `locomo` | `locomo10` | Multi-session conversation memory (LLM-judged) |
| `longmemeval` | `s`, `m` | Long-term memory QA (LLM-judged) |
| `memsim` | `simple`, `conditional`, `comparative`, … | Chinese personal memory MCQ |
| `membench` | `FirstAgentLowLevel`, `ThirdAgentHighLevel`, … | Agent trajectory memory (LLM-judged) |

## Memory providers

| Provider | Notes |
|---|---|
| `bm25` | Keyword search baseline, no API needed |
| `qdrant` | Vector search baseline, local Qdrant |
| `hindsight` | Embedded Hindsight — requires `GOOGLE_API_KEY` |
| `hindsight-cloud` | Hindsight cloud — requires `HINDSIGHT_CLOUD_KEY` |
| `mem0` | Local mem0 — requires `GOOGLE_API_KEY` |
| `mem0-cloud` | mem0 cloud — requires `MEM0_CLOUD_KEY` |
| `cognee` | Cognee knowledge graph |

---

## Contributing

### Submit a new result

Run the benchmark, compress and publish, then open a PR:

```bash
# 1. Run
uv run amb run --dataset personamem --split 32k --memory <your-provider>

# 2. Compress + upload to CDN
uv run amb publish-results outputs/personamem/<your-provider>/rag/32k.json --push

# 3. Commit and open a PR
git add outputs/ results-manifest.json .blob_manifest.json
git commit -m "results: <provider> on personamem 32k"
git push
```

A result file is a JSON at `outputs/{dataset}/{provider}/{mode}/{split}.json` (or `.json.gz`).
The `publish-results` command strips `raw_response` fields, gzips, and uploads to the CDN.

---

### Add a new memory provider

1. Create `src/memory_bench/memory/<name>.py` implementing `MemoryProvider`:

```python
from memory_bench.memory.base import MemoryProvider, RetrievedContext

class MyProvider(MemoryProvider):
    name = "my-provider"
    description = "One-line description"
    kind = "cloud"           # local | cloud | graph
    link = "https://..."
    logo = "https://..."

    def prepare(self, store_dir: Path) -> None:
        # Called before ingest/retrieve. Set up any persistent state here.
        ...

    async def ingest(self, docs: list[Document]) -> None:
        # Store documents in your memory system.
        ...

    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        # Return relevant context for the query.
        ...
```

2. Register it in `src/memory_bench/memory/__init__.py`:

```python
from .my_provider import MyProvider
REGISTRY["my-provider"] = MyProvider
```

3. Run on at least one dataset split and submit results (see above).

---

### Add a new dataset

1. Create `src/memory_bench/dataset/<name>.py` implementing `Dataset`:

```python
from memory_bench.dataset.base import Dataset, Document, Query

class MyDataset(Dataset):
    name = "my-dataset"
    description = "One-line description"
    task_type = "mcq"        # mcq | open
    splits = ["split1", "split2"]

    def load_documents(self, split: str) -> list[Document]:
        # Return list of Document(id=..., content=...)
        ...

    def load_queries(self, split: str) -> list[Query]:
        # Return list of Query(id=..., query=..., gold_answers=[...], gold_ids=[...])
        ...
```

2. Register it in `src/memory_bench/dataset/__init__.py`:

```python
from .my_dataset import MyDataset
REGISTRY["my-dataset"] = MyDataset
```

3. Export the dataset data and upload:

```bash
uv run amb publish-dataset --dataset my-dataset --push
git add data/ catalog.json
git commit -m "data: add my-dataset"
```

4. Run at least one provider on it and submit results.
