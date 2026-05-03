# Graph-RAG: Code-Aware Retrieval-Augmented Generation for Codebases

> **B.Tech Project (BTP)** — A graph-enhanced RAG pipeline that understands code structure, not just text similarity.

---

## Idea

Standard RAG systems treat source code as flat text — they split files into overlapping character windows, embed them, and retrieve the top-k chunks by vector similarity. This approach has three critical shortcomings when applied to codebases:

1. **Lost structural relationships** — A plain vector search doesn't know that `class Dog` inherits from `class Animal`, or that `process_data()` calls `validate_input()`. The LLM receives isolated snippets with no dependency context.
2. **Duplicate / noisy retrieval** — Naive chunking produces near-duplicate embeddings (e.g., a class chunk and its method chunk overlap heavily), wasting the limited context window.
3. **"Lost in the middle" effect** — Even when the right chunks are retrieved, burying them among irrelevant ones degrades LLM answer quality ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)).

**Graph-RAG** solves these by:
- Parsing source code with Python's `ast` module to extract **semantic chunks** (classes, functions, files) instead of arbitrary character windows.
- Building a **code-structure graph** where edges represent `calls`, `inherits`, and `called_by` relationships.
- Using **BFS graph traversal** after vector retrieval to walk the call-chain / inheritance-chain and pull in structurally related chunks the embedding model missed.
- Applying **hop-weighted token budgeting** — directly retrieved chunks (hop-0) get the most context, while graph-expanded chunks (hop-1, hop-2, …) get progressively less, keeping the LLM prompt focused.

The result: the LLM receives a coherent, dependency-aware view of the codebase — not just textually similar snippets.

---

## Architecture

```
┌─────────────┐      AST Parse       ┌──────────────────┐
│  Source Repo │  ─────────────────► │  Semantic Chunks │
│  (.py files) │                     │  + Rich Metadata │
└─────────────┘                      └────────┬─────────┘
                                              │
                               ┌──────────────┼──────────────┐
                               ▼              ▼              ▼
                         ┌──────────┐  ┌────────────┐ ┌──────────┐
                         │  FAISS   │  │  Code Graph│ │  Reverse │
                         │  Index   │  │  (calls,   │ │  Index   │
                         │ (vector) │  │  inherits) │ │called_by │
                         └────┬─────┘  └──────┬─────┘ └────┬─────┘
                              │               │            │
                              ▼               ▼            ▼
                    ┌──────────────────────────────────────────────┐
                    │            Hybrid Retrieval                  │
                    │  1. Vector top-k  →  keyword filter          │
                    │  2. BFS graph expansion (multi-hop)          │
                    │  3. Hop-weighted token budgeting             │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │   Qwen2.5-Coder LLM │
                                │   (1.5B, local CPU) │
                                └─────────────────────┘
```

---

## Folder Structure

```
PROJECT/
│
├── RAG/                            # Core RAG pipeline code
│   ├── baseline/
│   │   └── baseline_rag.py         # v0 — Naive char-window RAG (FAISS + flan-t5)
│   │
│   ├── improvements/
│   │   ├── v1_chunking.py          # v1 — AST-based semantic chunking
│   │   ├── v2_retrieval.py         # v2 — Keyword filtering + deduplication
│   │   ├── v3_content.py           # v3 — Context cleaning (noise removal)
│   │   ├── v4_metadata.py          # v4 — Rich metadata + 1-hop graph expansion
│   │   ├── v5_multihop.py          # v5 — Full multi-hop BFS graph traversal
│   │   ├── visualize_graph.py      # Interactive D3.js graph visualization
│   │   └── chunk_graph.html        # Generated graph output (open in browser)
│   │
│   └── README.md                   # ← You are here
│
├── data/                           # Target repositories to analyze (gitignored)
│   ├── sample_repo/                # Small test repo with inheritance patterns
│
│
├── evaluation/
│   └── test_query.txt              # Standard test queries for benchmarking
│
└── .gitignore
```

---

## Version Evolution

Each version builds on the previous one, addressing a specific limitation:

| Version | File | What Changed | Problem Solved |
|---------|------|-------------|----------------|
| **v0** (Baseline) | `baseline_rag.py` | Character-window chunking, flan-t5 seq2seq LLM | Starting point — flat text RAG |
| **v1** | `v1_chunking.py` | AST-based semantic chunking (class / function / file) | Eliminates mid-function splits and duplicate embeddings |
| **v2** | `v2_retrieval.py` | Keyword filtering, content deduplication, Qwen2.5-Coder LLM | Removes irrelevant chunks, proper instruction-following |
| **v3** | `v3_content.py` | Context cleaning — strips noise lines (print, logger, short lines) | Reduces token waste, keeps definitions + return statements |
| **v4** | `v4_metadata.py` | Rich AST metadata (calls, base_classes, methods, decorators, docstring, called_by), content-hash dedup, 1-hop graph expansion | LLM sees structural relationships, not just raw code |
| **v5** | `v5_multihop.py` | Configurable multi-hop BFS over the code graph, hop-weighted token budgeting | Full call-chain / inheritance-chain traversal (depth 1–N) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Code Parsing** | Python `ast` module |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Vector Store** | FAISS (IndexFlatL2) |
| **LLM** | Qwen2.5-Coder-1.5B-Instruct (runs on CPU, ~4-6 GB RAM) |
| **Graph Visualization** | D3.js v7 (self-contained HTML) |
| **Graph Traversal** | BFS with configurable depth + expansion cap |

---

## Getting Started

### Prerequisites

```bash
pip install sentence-transformers faiss-cpu transformers torch numpy
```

### Running

1. **Place your target codebase** in `data/sample_repo/` (or any subfolder under `data/`).

2. **Run the latest pipeline (v5)**:
   ```bash
   cd RAG/improvements
   python v5_multihop.py
   ```

3. **Visualize the code graph**:
   ```bash
   python visualize_graph.py
   # Open chunk_graph.html in your browser
   ```

4. **Run the baseline for comparison**:
   ```bash
   cd RAG/baseline
   python baseline_rag.py
   ```

### Configuring v5

The `query()` method accepts several parameters to tune retrieval:

```python
agent.query(
    "What classes inherit from a base class?",
    k=4,                # vector search top-k
    max_hops=3,         # BFS depth (0 = no graph, 1 = v4 equivalent)
    max_expansion=8,    # max extra chunks from graph traversal
    include_callers=False  # also follow called_by edges (upstream)
)
```

---

## Key Concepts

### Semantic Chunking (v1)
Instead of splitting code at arbitrary character boundaries, the AST parser extracts complete classes, functions, and file-level scripts as individual chunks. Methods inside classes are **not** separately indexed to avoid duplicate embeddings.

### Code-Structure Graph (v4–v5)
Each chunk carries metadata extracted from the AST:
- **`calls`** — functions/methods invoked inside this chunk
- **`base_classes`** — parent classes (inheritance)
- **`called_by`** — reverse index of callers (populated in a second pass)
- **`methods`** — methods defined in a class
- **`docstring`** — first docstring, truncated to 200 chars
- **`content_hash`** — MD5 of normalised source for deduplication

### Multi-Hop BFS (v5)
After vector retrieval returns the top-k chunks (hop-0), BFS traverses the graph edges:
- **Forward**: `calls` + `base_classes` edges
- **Backward** (opt-in): `called_by` edges
- Depth is configurable (`max_hops`, default 3)
- Expansion is capped (`max_expansion`, default 8)

### Hop-Weighted Token Budgeting (v5)
Chunks closer to the query get more of the LLM's context window:
- hop-0 → 400 chars
- hop-1 → 200 chars
- hop-2 → 100 chars
- hop-N → max(80, 400 >> N)

This prevents graph-expanded context from drowning out the directly relevant code.

---