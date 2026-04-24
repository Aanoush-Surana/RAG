import ast
import hashlib
import os
import re
import numpy as np
from collections import defaultdict

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
# AST HELPERS
# =============================================================================

def _get_docstring(node):
    """Extract the first docstring from a function or class node, if any."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value.strip()[:200]   # cap at 200 chars
    return ""


def _get_calls(node):
    """
    Return the set of function/method names directly called inside `node`.
    Only simple name calls (foo()) and attribute calls (self.foo(), obj.bar())
    are captured — we record the leaf name only, e.g. 'bar' for obj.bar().
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
    """Return the names of all direct methods of a class (not nested classes)."""
    methods = []
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(item.name)
    return methods


# =============================================================================
# CONTENT HASH  –  for deduplicating near-identical chunks across files
# =============================================================================
def _content_hash(code_text):
    """
    Produce a deterministic hash of normalised source code.
    Normalisation: collapse runs of whitespace, strip comments.
    Two chunks with the same hash are considered duplicates.
    """
    text = re.sub(r'#[^\n]*', '', code_text)    # strip inline comments
    text = re.sub(r'\s+', ' ', text).strip()     # collapse whitespace
    return hashlib.md5(text.encode()).hexdigest()


def _make_qname(file_path, name, repo_path=None):
    """
    Build a qualified name like  'modules/tracking/tracker.py::TrackerModule'
    so that same-named functions in different files are disambiguated.
    """
    if repo_path:
        rel = os.path.relpath(file_path, repo_path).replace("\\", "/")
    else:
        rel = os.path.basename(file_path)
    return f"{rel}::{name}"


# =============================================================================
# PARSER  –  produces ONE chunk per semantic unit with rich metadata
# =============================================================================
def parse_python_file(file_path, repo_path=None):
    """
    Returns a list of chunk dicts for a single .py file.

    Each chunk now carries:
      content       - raw source segment
      type          - 'class' | 'function' | 'file'
      file          - absolute file path
      name          - class/function/file name
      qname         - qualified name  (relative_path::name)
      content_hash  - md5 of normalised content (for dedup)
      base_classes  - [str]  (classes only)
      methods       - [str]  (classes only)
      calls         - [str]  names of functions called inside this chunk
      decorators    - [str]
      docstring     - first docstring, truncated to 200 chars
    """
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

    for node in tree.body:                          # top-level nodes only
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
                # ── structural metadata ──
                "base_classes": _get_base_classes(node),
                "methods":      _get_methods(node),
                "calls":        _get_calls(node),
                "decorators":   _get_decorators(node),
                "docstring":    _get_docstring(node),
                # reverse index filled later
                "called_by":    [],
                "also_in":      [],      # files where duplicate was found
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sub_chunks.append({
                "content":      segment,
                "type":         "function",
                "file":         file_path,
                "name":         node.name,
                "qname":        _make_qname(file_path, node.name, repo_path),
                "content_hash": _content_hash(segment),
                # ── structural metadata ──
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

    # Plain script with no top-level classes/functions
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
# CONTENT-HASH DEDUPLICATION
# =============================================================================
def dedup_chunks(chunks):
    """
    Remove near-identical chunks (same name + same content hash).
    When a duplicate is found, the *first* occurrence is kept and the
    duplicate's file path is recorded in `also_in` so we don't lose
    provenance information.

    Returns the deduplicated list.
    """
    seen = {}           # (name, content_hash) -> index in `unique`
    unique = []

    for chunk in chunks:
        key = (chunk["name"], chunk["content_hash"])
        if key in seen:
            # Record the duplicate's file for reference
            unique[seen[key]]["also_in"].append(
                os.path.basename(chunk["file"])
            )
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
# REPO LOADER
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

    # ── Deduplicate near-identical chunks ─────────────────────────────────
    all_chunks = dedup_chunks(all_chunks)

    return all_chunks


# =============================================================================
# REVERSE INDEX  –  populate `called_by` on every chunk (1-pass)
# =============================================================================
def build_reverse_index(chunks):
    """
    For every chunk B that chunk A calls, append A's name to B's `called_by`.
    This is done in-place so chunk dicts are mutated.

    Complexity: O(chunks × avg_calls) — typically fast for project-sized repos.
    """
    # name → list of chunk indices (same name may appear in multiple files)
    # Use qname (qualified) for precise resolution, but also index by bare
    # name so callee lookup (which only knows bare names) still works.
    qname_to_indices = defaultdict(list)
    name_to_indices  = defaultdict(list)
    for i, chunk in enumerate(chunks):
        qname_to_indices[chunk["qname"]].append(i)
        name_to_indices[chunk["name"]].append(i)

    for i, chunk in enumerate(chunks):
        caller_file = chunk["file"]
        caller_name = chunk["name"]
        for callee_name in chunk["calls"]:
            # Prefer same-file match, then fall back to any match
            candidates = name_to_indices[callee_name]
            same_file  = [j for j in candidates if chunks[j]["file"] == caller_file and j != i]
            targets    = same_file if same_file else [j for j in candidates if j != i]
            for j in targets:
                chunks[j]["called_by"].append(caller_name)

    # Deduplicate (a caller may appear many times if it's a large class)
    for chunk in chunks:
        chunk["called_by"] = sorted(set(chunk["called_by"]))

    print("[RAG] Reverse index built (called_by populated).")


# =============================================================================
# RAG AGENT
# =============================================================================
class RAGCodebaseAgent:

    CONTEXT_CHARS_PER_CHUNK = 400   # slightly larger now that headers carry info

    def __init__(self, repo_path):
        print(f"[RAG] Loading repo from: {repo_path}")

        self.documents = load_hierarchical_repo(repo_path)
        self.documents = [d for d in self.documents if len(d["content"]) > 80]

        if not self.documents:
            raise ValueError("No chunks found.")

        # ── Build reverse index (called_by) ─────────────────────────────────
        build_reverse_index(self.documents)

        # ── Qualified name + bare name indexes (for graph expansion) ────────
        self._qname_index = defaultdict(list)
        self._name_index  = defaultdict(list)
        for i, doc in enumerate(self.documents):
            self._qname_index[doc["qname"]].append(i)
            self._name_index[doc["name"]].append(i)

        self.chunks   = [d["content"] for d in self.documents]
        self.metadata = self.documents

        print(f"[RAG] {len(self.chunks)} chunks ready.")

        # ── ML Models ────────────────────────────────────────────────────────
        from sentence_transformers import SentenceTransformer
        import faiss
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Embed docstring + name + content together for richer semantic signal
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
    # v2: keyword filter (kept)
    # =========================================================================
    def keyword_filter(self, query, results):
        keywords = query.lower().split()
        filtered = [r for r in results if any(k in r[0].lower() for k in keywords)]
        return filtered if filtered else results

    # =========================================================================
    # v3: clean_chunk (kept — still useful for noisy bodies)
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
    # v4: graph expansion  –  1-hop: fetch base classes + callees
    # =========================================================================
    def _expand_with_graph(self, retrieved_indices):
        """
        Given a set of already-retrieved chunk indices, find 1-hop neighbors:
          • base_classes of class chunks
          • functions/classes in the `calls` list

        Returns a list of (chunk_content, meta) pairs for the new neighbors,
        capped so we don't overflow the LLM context.
        """
        expanded = []
        seen     = set(retrieved_indices)

        for idx in list(retrieved_indices):       # iterate over original set
            meta = self.metadata[idx]

            # Collect names to look up: base classes + called functions
            neighbors = meta["base_classes"] + meta["calls"]

            for name in neighbors:
                for neighbor_idx in self._name_index.get(name, []):
                    if neighbor_idx not in seen:
                        seen.add(neighbor_idx)
                        expanded.append((
                            self.chunks[neighbor_idx],
                            self.metadata[neighbor_idx],
                        ))
                        if len(expanded) >= 2:     # cap at 2 expansion chunks
                            return expanded

        return expanded

    # =========================================================================
    # Retrieve
    # =========================================================================
    def retrieve(self, query, k=4):
        q_emb = self.embed_model.encode([query])
        _, indices = self.index.search(np.array(q_emb, dtype="float32"), k)

        results          = []
        retrieved_indices = []
        seen             = set()

        print("\n--- RETRIEVED CHUNKS ---")
        for rank, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            meta  = self.metadata[idx]
            key   = chunk[:80]
            if key in seen:
                continue
            seen.add(key)
            retrieved_indices.append(idx)

            print(f"\n[{rank+1}] {meta['type'].upper()}  {meta['name']}  "
                  f"({os.path.basename(meta['file'])})")
            if meta["base_classes"]:
                print(f"     extends : {meta['base_classes']}")
            if meta["calls"]:
                print(f"     calls   : {meta['calls'][:5]}")
            if meta["called_by"]:
                print(f"     called_by: {meta['called_by'][:5]}")
            print(chunk[:200])
            results.append((chunk, meta))

        # keyword filter on the vector results
        results = self.keyword_filter(query, results)[:3]

        # ── v4: graph expansion ──────────────────────────────────────────────
        expansions = self._expand_with_graph(set(retrieved_indices))
        if expansions:
            print(f"\n[EXPAND] Adding {len(expansions)} graph-neighbor chunk(s):")
            for chunk, meta in expansions:
                print(f"  → {meta['type']} '{meta['name']}' from {os.path.basename(meta['file'])}")
            results.extend(expansions)

        return results

    # =========================================================================
    # Build enriched LLM header from metadata  (v4 key feature)
    # =========================================================================
    @staticmethod
    def _build_header(meta):
        """
        Produce a compact, information-dense header the LLM can parse.

        Example output:
            # class: DataProcessor  [file: processor.py]
            # extends : BaseProcessor, Mixin
            # methods : __init__, process, _validate
            # calls   : _validate, log_result
            # called_by: run_pipeline, main
            # doc: Processes incoming data frames using rolling windows.
        """
        lines = [f"# {meta['type']}: {meta['name']}  "
                 f"[file: {os.path.basename(meta['file'])}]"]

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
    # LLM call (unchanged from v3)
    # =========================================================================
    def run_llm(self, context, query):
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. "
                    "Answer questions about the codebase using ONLY the provided code context. "
                    "Use the structural annotations (extends, calls, called_by, methods) "
                    "to reason about relationships between components. "
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
    # Public query interface
    # =========================================================================
    def query(self, question, k=4):
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")

        retrieved = self.retrieve(question, k)

        # Build context: enriched header + cleaned body per chunk
        context_parts = []
        for chunk, meta in retrieved:
            header  = self._build_header(meta)
            cleaned = self.clean_chunk(chunk, question)
            body    = cleaned[:self.CONTEXT_CHARS_PER_CHUNK]
            context_parts.append(header + "\n" + body)

        context = "\n\n---\n\n".join(context_parts)

        print("\n--- CONTEXT SENT TO LLM ---")
        print(context[:800])

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

    queries = [
        "What does this project do?",
        "Explain the main architecture",
        "How is data processed?",
        "Which functions call the data processing logic?",
        "What classes inherit from a base class?",
        "Why is this approach used?",
    ]

    for q in queries:
        agent.query(q)
