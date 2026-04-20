import ast
import os
import numpy as np

# =============================================================================
# DIRECTORIES TO ALWAYS SKIP (noise / non-project code)
# =============================================================================
SKIP_DIRS = {
    "venv", ".venv", "env", ".env",
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", "site-packages",
    ".tox", "eggs", ".eggs",
}

# =============================================================================
# FILE PARSER  –  produces ONE chunk per semantic unit (no whole-file duplicate)
# =============================================================================
def parse_python_file(file_path):
    """
    Returns a list of chunks for a single .py file.

    Granularity strategy
    --------------------
    • If the file has classes/functions  →  emit those sub-units only.
      The whole-file blob is intentionally NOT added to avoid near-duplicate
      embeddings that pollute FAISS retrieval.
    • If the file has NO classes/functions (e.g. a pure script / config)
      →  fall back to one whole-file chunk so we don't silently lose it.
    • Methods inside a class are skipped at the top-level walk to avoid
      double-indexing them (the class chunk already contains them).
    """
    chunks = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except Exception as e:
        print(f"[ERROR] Cannot open {file_path}: {e}")
        return []

    # Skip very large generated / minified files
    if len(code) > 30_000:
        print(f"[SKIP] Too large ({len(code)//1000}k chars): {file_path}")
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        print(f"[SKIP] Syntax error, skipping: {file_path}")
        return []

    # Collect top-level classes and their method names so we can skip
    # bare FunctionDef nodes that are actually methods (already inside a class).
    class_method_names = set()
    top_level_nodes = []

    for node in tree.body:                      # only direct children of module
        if isinstance(node, ast.ClassDef):
            top_level_nodes.append(node)
            for item in ast.walk(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_method_names.add(id(item))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_nodes.append(node)

    sub_chunks = []

    for node in top_level_nodes:
        segment = ast.get_source_segment(code, node)
        if not segment or len(segment) < 60:   # skip trivial stubs
            continue

        if isinstance(node, ast.ClassDef):
            sub_chunks.append({
                "content": segment,
                "type": "class",
                "file": file_path,
                "name": node.name,
            })
        else:
            sub_chunks.append({
                "content": segment,
                "type": "function",
                "file": file_path,
                "name": node.name,
            })

    if sub_chunks:
        chunks.extend(sub_chunks)
    else:
        # Plain script with no classes/functions – keep it as a file chunk
        if len(code) >= 80:
            chunks.append({
                "content": code,
                "type": "file",
                "file": file_path,
                "name": os.path.basename(file_path),
            })

    return chunks


# =============================================================================
# REPO LOADER  –  walks only project source files
# =============================================================================
def load_hierarchical_repo(repo_path):
    """
    Walks `repo_path` and returns chunks for every project .py file,
    skipping venv / .git / __pycache__ and other noise directories.
    """
    all_chunks = []
    file_count = 0

    print(f"[RAG] Scanning repo: {repo_path}")

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # --- prune noise dirs IN PLACE so os.walk doesn't descend into them ---
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            rel_path  = os.path.relpath(file_path, repo_path)
            print(f"[RAG]   Parsing: {rel_path}")

            new_chunks = parse_python_file(file_path)
            all_chunks.extend(new_chunks)
            file_count += 1

    print(f"[RAG] Parsed {file_count} source files -> {len(all_chunks)} chunks")
    return all_chunks


# =============================================================================
# RAG AGENT
# =============================================================================
class RAGCodebaseAgent:

    # How many chars of context to feed the LLM per retrieved chunk.
    # flan-t5-base has a ~512-token encoder; ~300 chars ≈ 80-100 tokens,
    # so 4 chunks × 300 chars ≈ 350 tokens – comfortably within the limit.
    CONTEXT_CHARS_PER_CHUNK = 350

    def __init__(self, repo_path):
        print(f"[RAG] Loading repo from: {repo_path}")

        self.documents = load_hierarchical_repo(repo_path)
        self.documents = [d for d in self.documents if len(d["content"]) > 80]

        if not self.documents:
            raise ValueError("No chunks found. Check repo path and file filters.")

        self.chunks   = [d["content"] for d in self.documents]
        self.metadata = self.documents

        print(f"[RAG] {len(self.chunks)} chunks ready for embedding.")

        # ----- lazy imports (keeps startup fast before the models are needed) -----
        print("[RAG] Loading ML models (one-time, may take ~30 s)...")
        from sentence_transformers import SentenceTransformer
        import faiss
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embed_model.encode(self.chunks, show_progress_bar=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

        # ------------------------------------------------------------------ #
        # LLM: Qwen2.5-Coder-1.5B-Instruct                                   #
        # Why: flan-t5-base is a seq2seq summarisation model — it echoes its  #
        # input rather than answering questions.  Qwen2.5-Coder-1.5B-Instruct #
        # is a proper instruction-following causal LM tuned for code Q&A.     #
        # At 1.5 B parameters it runs on CPU in ~4-6 GB RAM.                  #
        # ------------------------------------------------------------------ #
        LLM_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        print(f"[RAG] Loading LLM: {LLM_ID}  (downloads once, ~3 GB) ...")

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            dtype=torch.float32,   # CPU-safe; use bfloat16 if you have a GPU
        )
        self.llm.eval()

        print("[RAG] Ready.\n")

    def keyword_filter(self, query, results):
        keywords = query.lower().split()
        filtered = []

        for chunk, meta in results:
            text = chunk.lower()

            if any(k in text for k in keywords):
                filtered.append((chunk, meta))

        return filtered if filtered else results

    # -------------------------------------------------------------------------
    def retrieve(self, query, k=4):
        """Return the k most relevant chunks for `query`."""
        q_emb = self.embed_model.encode([query])
        _, indices = self.index.search(np.array(q_emb, dtype="float32"), k)

        results = []
        seen    = set()                         # de-duplicate by content hash

        print("\n--- RETRIEVED CHUNKS ---")
        for rank, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            meta  = self.metadata[idx]
            key   = chunk[:80]                  # cheap content fingerprint
            if key in seen:
                continue
            seen.add(key)
            results.append((chunk, meta))
            print(f"\n[{rank+1}] {meta['type'].upper()}  {meta['name']}  "
                  f"({os.path.basename(meta['file'])})")
            print(chunk[:200])

        # Apply keyword filtering
        results = self.keyword_filter(query, results)

        # Reduce noise (top 3 only)
        results = results[:3]

        return results

    # -------------------------------------------------------------------------
    def run_llm(self, context, query):
        """
        Use Qwen2.5-Coder-Instruct chat template for a proper instruction-
        following answer.  The model is causal (decoder-only), so we pass
        input_ids and grab only the newly generated tokens.
        """
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. "
                    "Answer questions about the codebase using ONLY the provided code context. "
                    "Be concise and specific. Do NOT repeat the code verbatim."
                ),
            },
            {
                "role": "user",
                "content": f"Code context:\n{context}\n\nQuestion: {query}",
            },
        ]

        # Apply the chat template — this handles the special tokens properly
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
                do_sample=False,         # greedy is faster and more deterministic
                temperature=1.0,
                repetition_penalty=1.1,  # prevents the looping/echoing you saw
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        new_tokens = output_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # -------------------------------------------------------------------------
    def query(self, question, k=4):
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")

        retrieved = self.retrieve(question, k)

        # Build context: trim each chunk to CONTEXT_CHARS_PER_CHUNK chars
        context_parts = []
        for chunk, meta in retrieved:
            header = f"# {meta['type']}: {meta['name']} ({os.path.basename(meta['file'])})\n"
            body   = chunk[:self.CONTEXT_CHARS_PER_CHUNK]
            context_parts.append(header + body)

        context = "\n\n---\n\n".join(context_parts)

        print("\n--- CONTEXT SENT TO LLM ---")
        print(context[:600])

        answer = self.run_llm(context, question)

        print("\n--- MODEL OUTPUT ---")
        print(answer)
        return answer


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    repo_path = r"C:\Users\Aanoush Surana\OneDrive\Desktop\BTP\data\sample_repo"
    agent = RAGCodebaseAgent(repo_path)

    queries = [
        "What does this project do?",
        "How is data processed?",
        "Explain the main architecture",
        "How does the system store data?",
        "Why is this approach used?",
    ]

    for q in queries:
        agent.query(q)