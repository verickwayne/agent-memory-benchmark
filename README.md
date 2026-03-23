# OMB — Open Memory Benchmark

A modular benchmark runner for evaluating memory systems on personal and long-horizon QA tasks.

## How it works

1. **Ingest** — documents from a dataset are loaded into a memory provider
2. **Retrieve** — for each query the memory provider retrieves relevant context
3. **Generate** — a Gemini model produces an answer from the retrieved context
4. **Judge** — a second Gemini call scores the answer against gold answers

Retrieval time is tracked separately from generation; ingestion time is also recorded.

## Setup

```bash
# Copy and fill in your API key
cp .env.example .env   # or just create .env with:
# GEMINI_API_KEY=...
```

## Usage

```bash
# List available datasets, memory providers, and modes
omb providers

# List domains for a dataset
omb domains --dataset personamem

# Run a benchmark
omb run --dataset personamem --domain 32k --memory bm25

# Limit scale for a quick test
omb run --dataset personamem --domain 32k --memory bm25 --query-limit 20

# Oracle mode: ingest only gold documents (tests generation quality in isolation)
omb run --dataset personamem --domain 32k --memory bm25 --oracle

# Dataset statistics
omb dataset-stats --dataset personamem

# Browse results in the browser
omb view
```

## Datasets

| Dataset | Domains | Task |
|---|---|---|
| `tempo` | 13 domains (bitcoin, law, politics, …) | Factual retrieval QA |
| `memsim` | simple, conditional, comparative, aggregative, post_processing, noisy | Chinese personal memory MCQ |
| `personamem` | 32k, 128k, 1M | Long-horizon preference tracking MCQ |
| `membench` | FirstAgentLowLevel/HighLevel, ThirdAgentLowLevel/HighLevel | Requires local data download |

## Memory providers

| Provider | Description |
|---|---|
| `bm25` | In-memory BM25 (reference baseline) |
| `mem0` | Mem0 with Gemini LLM + embedder, local Qdrant |

## Results

Results are saved to `outputs/{dataset}/{memory}/{mode}/{domain}.json` and can be explored with `omb view`.

## Requirements

- Python ≥ 3.11
- `GEMINI_API_KEY` in `.env` or environment
- For MemBench: set `MEMBENCH_DATA_PATH` to your local data directory


