"""
v5_multihop.py  –  Code-Aware RAG with Full Multi-Hop Graph Traversal
======================================================================
Builds on v4_metadata.py and replaces the fixed 1-hop graph expansion
with a configurable BFS that follows the complete call-chain /
inheritance chain to an arbitrary depth.

Key additions over v4
─────────────────────
• _expand_with_graph_multihop()
    – BFS over (base_classes ∪ calls) edges, optionally (called_by) too
    – Configurable max_hops  (default 3)
    – Configurable max_expansion  (default 8 extra chunks)
    – Returns (content, meta, hop_distance) triples

• retrieve() returns (chunk, meta, hop) triples
    – hop = 0  →  retrieved directly by vector search
    – hop ≥ 1  →  reached by graph traversal

• _build_header() shows [hop=N] so the LLM knows provenance distance

• query() applies hop-weighted token budgeting
    – hop-0 chunks get CONTEXT_CHARS_PER_CHUNK chars
    – each extra hop halves the budget (floor at MIN_CHARS_PER_CHUNK)
    – keeps the LLM prompt focused on the most relevant code
"""

import ast
import hashlib
import os
import re
from collections import defaultdict, deque

import numpy as np

# =============================================================================
# DIRECTORIES TO SKIP
# =============================================================================
SKIP_DIRS = {
    "venv", ".venv", "env", ".env",
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", "site-packages",
    ".tox", "eggs", ".eggs",
}

# =============================================================================
# AST HELPERS  (unchanged from v4)
# =============================================================================

def _get_docstring(node):
    """Extract the first docstring from a function or class node, if any."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value.strip()[:200]
    return ""


def _get_calls(node):
    """
    Return the sorted set of function/method names directly called inside `node`.
    Captures:
      • Simple name calls:  foo()       -> 'foo'
      • Attribute calls:    obj.bar()   -> 'bar'
      • Class instantiations: Bat("x") -> 'Bat'
    """
    calls = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
    return sorted(calls)


def _get_decorators(node):
    """Return a list of decorator name strings on a function/class node."""
    names = []
    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(dec.attr)
        elif isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


def _get_base_classes(class_node):
    """Return the list of base-class names for an ast.ClassDef node."""
    bases = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(base.attr)
    return bases


def _get_methods(class_node):
    """Return the names of all direct methods of a class."""
    return [
        item.name
        for item in class_node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


# =============================================================================
# CONTENT HASH  (unchanged from v4)
# =============================================================================
def _content_hash(code_text):
    text = re.sub(r'#[^\n]*', '', code_text)
    text = re.sub(r'\s+', ' ', text).strip()
    return hashlib.md5(text.encode()).hexdigest()


def _make_qname(file_path, name, repo_path=None):
    if repo_path:
        rel = os.path.relpath(file_path, repo_path).replace("\\", "/")
    else:
        rel = os.path.basename(file_path)
    return f"{rel}::{name}"


# =============================================================================
# PARSER  (unchanged from v4)
# =============================================================================
def parse_python_file(file_path, repo_path=None):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except Exception as e:
        print(f"[ERROR] Cannot open {file_path}: {e}")
        return []

    if len(code) > 30_000:
        print(f"[SKIP] Too large ({len(code)//1000}k chars): {file_path}")
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        print(f"[SKIP] Syntax error: {file_path}")
        return []

    sub_chunks = []

    for node in tree.body:
        segment = ast.get_source_segment(code, node)
        if not segment or len(segment) < 60:
            continue

        if isinstance(node, ast.ClassDef):
            sub_chunks.append({
                "content":      segment,
                "type":         "class",
                "file":         file_path,
                "name":         node.name,
                "qname":        _make_qname(file_path, node.name, repo_path),
                "content_hash": _content_hash(segment),
                "base_classes": _get_base_classes(node),
                "methods":      _get_methods(node),
                "calls":        _get_calls(node),
                "decorators":   _get_decorators(node),
                "docstring":    _get_docstring(node),
                "called_by":    [],
                "also_in":      [],
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sub_chunks.append({
                "content":      segment,
                "type":         "function",
                "file":         file_path,
                "name":         node.name,
                "qname":        _make_qname(file_path, node.name, repo_path),
                "content_hash": _content_hash(segment),
                "base_classes": [],
                "methods":      [],
                "calls":        _get_calls(node),
                "decorators":   _get_decorators(node),
                "docstring":    _get_docstring(node),
                "called_by":    [],
                "also_in":      [],
            })

    if sub_chunks:
        return sub_chunks

    return [{
        "content":      code,
        "type":         "file",
        "file":         file_path,
        "name":         os.path.basename(file_path),
        "qname":        _make_qname(file_path, os.path.basename(file_path), repo_path),
        "content_hash": _content_hash(code),
        "base_classes": [],
        "methods":      [],
        "calls":        _get_calls(tree),
        "decorators":   [],
        "docstring":    "",
        "called_by":    [],
        "also_in":      [],
    }]


# =============================================================================
# CONTENT-HASH DEDUPLICATION  (unchanged from v4)
# =============================================================================
def dedup_chunks(chunks):
    seen   = {}
    unique = []

    for chunk in chunks:
        key = (chunk["name"], chunk["content_hash"])
        if key in seen:
            unique[seen[key]]["also_in"].append(os.path.basename(chunk["file"]))
            print(f"[DEDUP] Dropped duplicate: {chunk['name']} "
                  f"(from {os.path.basename(chunk['file'])}, "
                  f"kept {os.path.basename(unique[seen[key]]['file'])})")
        else:
            seen[key] = len(unique)
            unique.append(chunk)

    if len(chunks) != len(unique):
        print(f"[DEDUP] {len(chunks)} -> {len(unique)}  "
              f"({len(chunks) - len(unique)} duplicates removed)")
    return unique


# =============================================================================
# REPO LOADER  (unchanged from v4)
# =============================================================================
def load_hierarchical_repo(repo_path):
    all_chunks = []
    file_count = 0
    print(f"[RAG] Scanning repo: {repo_path}")

    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = os.path.join(root, file)
            rel_path  = os.path.relpath(file_path, repo_path)
            print(f"[RAG]   Parsing: {rel_path}")
            all_chunks.extend(parse_python_file(file_path, repo_path))
            file_count += 1

    print(f"[RAG] Parsed {file_count} files -> {len(all_chunks)} chunks")
    all_chunks = dedup_chunks(all_chunks)
    return all_chunks


# =============================================================================
# REVERSE INDEX  (unchanged from v4)
# =============================================================================
def build_reverse_index(chunks):
    name_to_indices   = defaultdict(list)
    method_to_classes = defaultdict(list)

    for i, chunk in enumerate(chunks):
        name_to_indices[chunk["name"]].append(i)
        if chunk["type"] == "class":
            for method in chunk.get("methods", []):
                method_to_classes[method].append(i)

    for i, chunk in enumerate(chunks):
        caller_file = chunk["file"]
        resolved    = set()

        for callee_name in chunk["calls"]:
            candidates = name_to_indices[callee_name]
            same_file  = [j for j in candidates
                          if chunks[j]["file"] == caller_file and j != i]
            targets    = same_file if same_file else [j for j in candidates if j != i]

            if not targets:
                owners     = method_to_classes[callee_name]
                same_file2 = [j for j in owners
                               if chunks[j]["file"] == caller_file and j != i]
                targets    = same_file2 if same_file2 else [j for j in owners if j != i]

            for j in targets:
                if j not in resolved:
                    resolved.add(j)
                    chunks[j]["called_by"].append(chunk["name"])

    for chunk in chunks:
        chunk["called_by"] = sorted(set(chunk["called_by"]))

    print("[RAG] Reverse index built (called_by populated).")


# =============================================================================
# RAG AGENT  –  v5: multi-hop graph traversal
# =============================================================================
class RAGCodebaseAgent:

    # Token budget (chars) for hop-0 (directly retrieved) chunks
    CONTEXT_CHARS_PER_CHUNK = 400
    # Floor: even the deepest hop gets at least this many chars
    MIN_CHARS_PER_CHUNK = 80

    def __init__(self, repo_path):
        print(f"[RAG] Loading repo from: {repo_path}")

        self.documents = load_hierarchical_repo(repo_path)
        self.documents = [d for d in self.documents if len(d["content"]) > 80]

        if not self.documents:
            raise ValueError("No chunks found.")

        build_reverse_index(self.documents)

        self._qname_index = defaultdict(list)
        self._name_index  = defaultdict(list)
        for i, doc in enumerate(self.documents):
            self._qname_index[doc["qname"]].append(i)
            self._name_index[doc["name"]].append(i)

        self.chunks   = [d["content"] for d in self.documents]
        self.metadata = self.documents

        print(f"[RAG] {len(self.chunks)} chunks ready.")

        from sentence_transformers import SentenceTransformer
        import faiss
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        embed_texts = [
            f"{d['name']}: {d['docstring']}\n{d['content']}"
            for d in self.documents
        ]
        embeddings = self.embed_model.encode(embed_texts, show_progress_bar=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

        LLM_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            torch_dtype=torch.float32,
        )
        self.llm.eval()
        print("[RAG] Ready.\n")

    # =========================================================================
    # v2: keyword filter  (unchanged)
    # =========================================================================
    def keyword_filter(self, query, results):
        """Filter (chunk, meta, hop) triples by query keywords."""
        keywords = query.lower().split()
        filtered = [r for r in results if any(k in r[0].lower() for k in keywords)]
        return filtered if filtered else results

    # =========================================================================
    # v3: clean_chunk  (unchanged)
    # =========================================================================
    def clean_chunk(self, chunk, query):
        lines    = chunk.split("\n")
        keywords = query.lower().split()
        cleaned  = []

        for line in lines:
            l = line.strip()
            if l.startswith("def ") or l.startswith("class "):
                cleaned.append(l); continue
            if any(k in l.lower() for k in keywords):
                cleaned.append(l); continue
            if l.startswith("return"):
                cleaned.append(l); continue
            if l.startswith("print") or l.startswith("logger"):
                continue
            if len(l) < 8:
                continue

        return "\n".join(cleaned) if len(cleaned) >= 3 else "\n".join(lines[:10])

    # =========================================================================
    # v5: MULTI-HOP BFS graph expansion  ← core new feature
    # =========================================================================
    def _expand_with_graph_multihop(
        self,
        retrieved_indices,
        max_hops: int = 3,
        max_expansion: int = 8,
        include_callers: bool = False,
    ):
        """
        BFS graph expansion over the code-structure graph.

        Starting from `retrieved_indices` (hop 0), we traverse edges:
          • Forward  : base_classes  (inheritance chain upward)
          • Forward  : calls         (call-chain downward)
          • Backward : called_by     (call-chain upward, opt-in via include_callers)

        Parameters
        ----------
        retrieved_indices : set[int]
            Chunk indices already returned by the vector retriever (hop 0).
        max_hops : int
            BFS depth limit.
              1  → original v4 behaviour
              3  → recommended (full chain for most real patterns)
              0  → disables graph expansion entirely
        max_expansion : int
            Hard cap on how many *new* chunks we add.  Prevents token overflow.
        include_callers : bool
            If True, also follow called_by edges so upstream callers are pulled in.
            Useful for "who uses this?" queries but can add noise for "what does
            X do?" queries.  Defaults to False.

        Returns
        -------
        list of (chunk_content, metadata_dict, hop_distance) triples,
        ordered by ascending hop distance (closest neighbours first).
        """
        if max_hops == 0:
            return []

        # BFS queue entries: (chunk_index, hop_distance)
        frontier: deque = deque()
        seen: set       = set(retrieved_indices)
        expanded        = []          # (content, meta, hop)

        # Seed the frontier with every directly-retrieved chunk
        for idx in retrieved_indices:
            frontier.append((idx, 0))

        while frontier and len(expanded) < max_expansion:
            idx, hop = frontier.popleft()

            # Don't expand beyond the depth limit
            if hop >= max_hops:
                continue

            meta = self.metadata[idx]

            # ── Collect neighbour names ────────────────────────────────────
            neighbor_names = list(meta["base_classes"]) + list(meta["calls"])
            if include_callers:
                neighbor_names += list(meta["called_by"])

            # ── Resolve names → chunk indices ──────────────────────────────
            for name in neighbor_names:
                for neighbor_idx in self._name_index.get(name, []):
                    if neighbor_idx in seen:
                        continue

                    seen.add(neighbor_idx)
                    next_hop = hop + 1

                    expanded.append((
                        self.chunks[neighbor_idx],
                        self.metadata[neighbor_idx],
                        next_hop,
                    ))

                    # Keep exploring from this new node (unless cap hit)
                    if len(expanded) < max_expansion:
                        frontier.append((neighbor_idx, next_hop))

                    if len(expanded) >= max_expansion:
                        break

                if len(expanded) >= max_expansion:
                    break

        return expanded

    # =========================================================================
    # Retrieve  –  returns (chunk, meta, hop) triples
    # =========================================================================
    def retrieve(self, query, k=4, max_hops=3, max_expansion=8,
                 include_callers=False):
        """
        Parameters
        ----------
        k              : number of vector-search results
        max_hops       : BFS depth for graph expansion (0 disables it)
        max_expansion  : max extra chunks added via graph traversal
        include_callers: also follow called_by edges during BFS
        """
        q_emb = self.embed_model.encode([query])
        _, indices = self.index.search(np.array(q_emb, dtype="float32"), k)

        results           = []    # (chunk, meta, hop=0)
        retrieved_indices = []
        seen_keys         = set()

        print("\n--- RETRIEVED CHUNKS (vector search) ---")
        for rank, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            meta  = self.metadata[idx]
            key   = chunk[:80]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            retrieved_indices.append(idx)

            print(f"\n[hop=0 | rank={rank+1}] {meta['type'].upper()}  {meta['name']}  "
                  f"({os.path.basename(meta['file'])})")
            if meta["base_classes"]:
                print(f"     extends : {meta['base_classes']}")
            if meta["calls"]:
                print(f"     calls   : {meta['calls'][:5]}")
            if meta["called_by"]:
                print(f"     called_by: {meta['called_by'][:5]}")
            print(chunk[:200])

            results.append((chunk, meta, 0))   # ← hop=0

        # Keyword filter (operates on the hop-0 set only, preserving hop field)
        results = self.keyword_filter(query, results)[:3]

        # ── v5: multi-hop BFS expansion ──────────────────────────────────────
        if max_hops > 0:
            expansions = self._expand_with_graph_multihop(
                set(retrieved_indices),
                max_hops=max_hops,
                max_expansion=max_expansion,
                include_callers=include_callers,
            )

            if expansions:
                print(f"\n[EXPAND] Multi-hop BFS added {len(expansions)} chunk(s) "
                      f"(max_hops={max_hops}):")
                for chunk, meta, hop in expansions:
                    print(f"  [hop={hop}] {meta['type']} '{meta['name']}'"
                          f"  ({os.path.basename(meta['file'])})")
                results.extend(expansions)

        return results   # list of (chunk, meta, hop)

    # =========================================================================
    # Build enriched LLM header  –  now shows hop distance
    # =========================================================================
    @staticmethod
    def _build_header(meta, hop: int = 0):
        """
        Compact header for the LLM.  Shows hop distance so the model knows
        whether this chunk was directly relevant (hop=0) or pulled in via
        graph traversal (hop≥1).
        """
        hop_label = "direct" if hop == 0 else f"hop-{hop}"
        lines = [
            f"# {meta['type']}: {meta['name']}  "
            f"[file: {os.path.basename(meta['file'])}]  [retrieval: {hop_label}]"
        ]

        if meta.get("base_classes"):
            lines.append(f"# extends  : {', '.join(meta['base_classes'])}")
        if meta.get("methods"):
            lines.append(f"# methods  : {', '.join(meta['methods'][:8])}")
        if meta.get("calls"):
            lines.append(f"# calls    : {', '.join(meta['calls'][:6])}")
        if meta.get("called_by"):
            lines.append(f"# called_by: {', '.join(meta['called_by'][:5])}")
        if meta.get("decorators"):
            lines.append(f"# decorators: {', '.join(meta['decorators'])}")
        if meta.get("docstring"):
            lines.append(f"# doc: {meta['docstring'][:120]}")
        if meta.get("also_in"):
            lines.append(f"# also_in: {', '.join(meta['also_in'])}")

        return "\n".join(lines)

    # =========================================================================
    # LLM call  (unchanged from v4)
    # =========================================================================
    def run_llm(self, context, query):
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. "
                    "Answer questions about the codebase using ONLY the provided code context. "
                    "Use the structural annotations (extends, calls, called_by, methods, retrieval) "
                    "to reason about relationships between components. "
                    "Chunks marked [retrieval: direct] are the most relevant; "
                    "chunks marked [retrieval: hop-N] provide supporting context. "
                    "Be concise and specific. Do NOT repeat the code verbatim."
                ),
            },
            {
                "role": "user",
                "content": f"Code context:\n{context}\n\nQuestion: {query}",
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to("cpu")

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.llm.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # =========================================================================
    # Hop-weighted token budget
    # =========================================================================
    def _char_budget(self, hop: int) -> int:
        """
        Allocate fewer characters to chunks reached by deeper hops.

        hop 0  →  CONTEXT_CHARS_PER_CHUNK        (e.g. 400)
        hop 1  →  CONTEXT_CHARS_PER_CHUNK // 2   (200)
        hop 2  →  CONTEXT_CHARS_PER_CHUNK // 4   (100)
        hop N  →  max(MIN_CHARS_PER_CHUNK, budget // 2^N)

        This keeps the prompt focused on directly-relevant code while still
        giving the LLM enough context from the dependency chain to reason
        about cross-component relationships.
        """
        budget = self.CONTEXT_CHARS_PER_CHUNK >> hop   # right-shift = integer halving
        return max(budget, self.MIN_CHARS_PER_CHUNK)

    # =========================================================================
    # Public query interface
    # =========================================================================
    def query(self, question, k=4, max_hops=3, max_expansion=8,
              include_callers=False):
        """
        Parameters
        ----------
        question        : natural-language question about the codebase
        k               : vector-search top-k
        max_hops        : BFS depth (0 = no graph expansion, 1 = v4 behaviour)
        max_expansion   : max graph-expanded chunks to add
        include_callers : also pull in upstream callers during BFS
        """
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print(f"CONFIG: k={k}  max_hops={max_hops}  max_expansion={max_expansion}"
              f"  include_callers={include_callers}")

        retrieved = self.retrieve(
            question, k=k,
            max_hops=max_hops,
            max_expansion=max_expansion,
            include_callers=include_callers,
        )

        # Build context with hop-weighted character budgets
        context_parts = []
        for chunk, meta, hop in retrieved:
            header  = self._build_header(meta, hop)
            cleaned = self.clean_chunk(chunk, question)
            budget  = self._char_budget(hop)
            body    = cleaned[:budget]
            context_parts.append(header + "\n" + body)

        context = "\n\n---\n\n".join(context_parts)

        print("\n--- CONTEXT SENT TO LLM ---")
        print(context[:1000])

        answer = self.run_llm(context, question)

        print("\n--- MODEL OUTPUT ---")
        print(answer)
        return answer


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    repo_path = r"C:\Users\Aanoush Surana\OneDrive\Desktop\BTP\data\sample_repo"

    agent = RAGCodebaseAgent(repo_path)

    # ── Example 1: default multi-hop (depth 3, forward edges only) ──────────
    print("\n" + "="*60)
    print("DEMO 1 — full call-chain traversal (max_hops=3)")
    agent.query(
        "What classes inherit from a base class and what do they call?",
        max_hops=3,
        max_expansion=8,
    )

    # ── Example 2: include upstream callers too ──────────────────────────────
    print("\n" + "="*60)
    print("DEMO 2 — forward + backward traversal (include_callers=True)")
    agent.query(
        "Which functions call the data processing logic?",
        max_hops=3,
        max_expansion=8,
        include_callers=True,
    )

    # ── Example 3: disable graph expansion (pure vector RAG baseline) ────────
    print("\n" + "="*60)
    print("DEMO 3 — no graph expansion (max_hops=0, baseline)")
    agent.query(
        "How is data processed?",
        max_hops=0,
    )

    # ── Example 4: shallow single-hop (v4 equivalent) ────────────────────────
    print("\n" + "="*60)
    print("DEMO 4 — 1-hop only (v4 equivalent)")
    agent.query(
        "Explain the main architecture",
        max_hops=1,
        max_expansion=2,
    )
