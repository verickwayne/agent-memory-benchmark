# AMA-Bench Dataset Reference

**Links:**
- GitHub: https://github.com/AMA-Bench/AMA-Bench
- Website: https://ama-bench.github.io/
- Paper (ICLR 2026 Memory Agent Workshop): https://arxiv.org/html/2602.22769v2
- HuggingFace: https://huggingface.co/datasets/AMA-bench/AMA-bench

---

## Overview

AMA-Bench (Evaluating Long-Horizon Memory for Agentic Applications) tests whether a system can build memory from long agent trajectories and answer questions about them. It targets real-world agentic use cases: SQL agents, web agents, gaming, software engineering, embodied AI.

**Task type:** Open-ended QA (free-text answers, LLM-as-judge scoring)
> Note: An MCQ subset is referenced in the codebase (`mcq_set.jsonl`) but not yet publicly released. Only open-ended is available.

---

## HuggingFace Dataset

```python
from datasets import load_dataset
ds = load_dataset("AMA-bench/AMA-bench", split="test")
# 208 rows, single "test" split
```

- **Config:** `default`
- **Split:** `test` only (208 episodes)
- **File:** `test/open_end_qa_set.jsonl` (~50 MB)
- **Total QA pairs:** 208 × 12 = **2,496 queries**

---

## Schema

### Row fields

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | int | Sequential ID (0-indexed) |
| `task` | string | Natural language task description |
| `domain` | string | One of 6 domains (see below) |
| `task_type` | string | Fine-grained category (e.g. `babaisai`, `crafter`) |
| `success` | bool | Whether the agent completed the task |
| `num_turns` | int | Number of trajectory turns (range: 9–525) |
| `total_tokens` | int | Tokens in episode (range: 315–1,030,000) |
| `trajectory` | list | Sequence of agent-environment turns |
| `qa_pairs` | list | 12 memory-testing QA pairs per episode |

### `trajectory` element

```json
{
  "turn_idx": 0,
  "action": "agent action taken",
  "observation": "environment response"
}
```

Flattened to text:
```
# Task
{task description}

# Agent Trajectory
Step 0:
  Action: {action}
  Observation: {observation}
...
```

### `qa_pairs` element

```json
{
  "question": "natural language question about the trajectory",
  "answer": "ground-truth free-text answer",
  "question_uuid": "uuid4",
  "type": "A" | "B" | "C" | "D"
}
```

---

## Domains (6)

| Domain | Episodes | QA Pairs |
|--------|----------|----------|
| Text-to-SQL | 51 | 612 |
| Open-World Tool QA | 30 | 360 |
| Web Task Execution | 31 | 372 |
| Gaming | 30 | 360 |
| Embodied AI | 30 | 360 |
| Software Engineering | 36 | 432 |

---

## Question Types (categories)

| Type | Capability | Description |
|------|-----------|-------------|
| `A` | Recall | Temporal/sequential information identification |
| `B` | Causal Inference | Action preconditions and state dependency |
| `C` | State Updating | Tracking explicit/hidden state changes |
| `D` | State Abstraction | Redundancy filtering, information extraction |

Distribution: A=839, B=596, C=647, D=414 pairs.

---

## Evaluation

- **Judge:** LLM-as-judge (paper uses Qwen3-32B; our bench uses Gemini)
- **Score:** Binary 1/0 per question (yes/no from judge)
- **Metrics:** Overall accuracy + breakdown by type (A/B/C/D), domain, task_type
- The judge receives: question + ground-truth answer + predicted answer → yes/no

---

## Implementation Plan

### Document mapping
- One document per episode (`episode_id` → doc ID)
- Content = flattened trajectory text (prepended with task description)
- Metadata: `domain`, `task_type`, `success`, `num_turns`, `total_tokens`

### Query mapping
- One query per QA pair (12 per episode → 2,496 total)
- Query ID = `question_uuid`
- Query text = `question`
- Gold answer = `answer`
- Gold doc ID = `episode_id`
- Metadata: `type` (A/B/C/D), `domain`, `task_type`

### Split / category design (proposed)
- **Split:** `test` (the only available split)
- **Category:** `type` field (A/B/C/D) — allows targeted evaluation by memory capability

### Key consideration
Trajectories average ~57K tokens (real-world subset). Most memory providers will need chunking or summarization strategies. The benchmark specifically stresses long-context compression.
