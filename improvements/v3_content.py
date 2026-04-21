import ast
import os
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
# PARSER
# =============================================================================
def parse_python_file(file_path):
    chunks = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except:
        return []

    if len(code) > 30000:
        print(f"[SKIP] Too large: {file_path}")
        return []

    try:
        tree = ast.parse(code)
    except:
        return []

    sub_chunks = []

    for node in tree.body:
        segment = ast.get_source_segment(code, node)

        if not segment or len(segment) < 60:
            continue

        if isinstance(node, ast.ClassDef):
            sub_chunks.append({
                "content": segment,
                "type": "class",
                "file": file_path,
                "name": node.name,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sub_chunks.append({
                "content": segment,
                "type": "function",
                "file": file_path,
                "name": node.name,
            })

    if sub_chunks:
        return sub_chunks

    return [{
        "content": code,
        "type": "file",
        "file": file_path,
        "name": os.path.basename(file_path),
    }]


# =============================================================================
# LOADER
# =============================================================================
def load_hierarchical_repo(repo_path):
    all_chunks = []

    print(f"[RAG] Scanning repo: {repo_path}")

    for root, dirs, files in os.walk(repo_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            print(f"[RAG] Parsing: {file}")

            all_chunks.extend(parse_python_file(file_path))

    print(f"[RAG] Total chunks: {len(all_chunks)}")
    return all_chunks


# =============================================================================
# RAG AGENT
# =============================================================================
class RAGCodebaseAgent:

    CONTEXT_CHARS_PER_CHUNK = 350

    def __init__(self, repo_path):
        print(f"[RAG] Loading repo from: {repo_path}")

        self.documents = load_hierarchical_repo(repo_path)
        self.documents = [d for d in self.documents if len(d["content"]) > 80]

        if not self.documents:
            raise ValueError("No chunks found.")

        self.chunks = [d["content"] for d in self.documents]
        self.metadata = self.documents

        print(f"[RAG] {len(self.chunks)} chunks ready.")

        from sentence_transformers import SentenceTransformer
        import faiss
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embed_model.encode(self.chunks)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

        LLM_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            torch_dtype=torch.float32,   # CPU-safe; use bfloat16 if you have a GPU
        )
        self.llm.eval()

    # =========================
    # v2: keyword filter
    # =========================
    def keyword_filter(self, query, results):
        keywords = query.lower().split()
        filtered = []

        for chunk, meta in results:
            if any(k in chunk.lower() for k in keywords):
                filtered.append((chunk, meta))

        return filtered if filtered else results

    # =========================
    # v3: CLEAN CONTEXT
    # =========================
    def clean_chunk(self, chunk, query):
        lines = chunk.split("\n")
        keywords = query.lower().split()

        cleaned = []

        for line in lines:
            l = line.strip()

            if l.startswith("def ") or l.startswith("class "):
                cleaned.append(l)
                continue

            if any(k in l.lower() for k in keywords):
                cleaned.append(l)
                continue

            if l.startswith("return"):
                cleaned.append(l)
                continue

            if ("import" in l or "from" in l) and any(k in l.lower() for k in keywords):
                cleaned.append(l)
                continue

            if l.startswith("print") or l.startswith("logger"):
                continue

            if len(l) < 8:
                continue

        if len(cleaned) < 3:
            cleaned = lines[:10]

        return "\n".join(cleaned)

    # =========================
    def retrieve(self, query, k=4):
        q_emb = self.embed_model.encode([query])
        _, indices = self.index.search(np.array(q_emb, dtype="float32"), k)

        results = []
        seen = set()                        # de-duplicate by content fingerprint
        print("\n--- RETRIEVED CHUNKS ---")

        for i, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            meta = self.metadata[idx]
            key = chunk[:80]                # cheap content fingerprint

            if key in seen:
                continue
            seen.add(key)

            print(f"\n[{i+1}] {meta['type']} {meta['name']} ({meta['file']})")
            print(chunk[:200])

            results.append((chunk, meta))

        results = self.keyword_filter(query, results)
        results = results[:3]

        return results

    # =========================
    def run_llm(self, context, query):
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

        # Chat template handles Qwen's special tokens correctly
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
                do_sample=False,          # greedy — faster and deterministic
                temperature=1.0,
                repetition_penalty=1.1,   # prevents prompt echoing
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        new_tokens = output_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # =========================
    def query(self, question, k=4):
        print(f"\n{'='*50}")
        print("QUERY:", question)

        retrieved = self.retrieve(question, k)

        # 🔥 CLEANED CONTEXT (v3)
        context_parts = []

        for chunk, meta in retrieved:
            header = f"# {meta['type']}: {meta['name']}\n"

            cleaned = self.clean_chunk(chunk, question)
            body = cleaned[:self.CONTEXT_CHARS_PER_CHUNK]

            context_parts.append(header + body)

        context = "\n\n---\n\n".join(context_parts)

        print("\n--- CLEANED CONTEXT ---")
        print(context[:500])

        answer = self.run_llm(context, question)

        print("\n--- OUTPUT ---")
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
        "Why is this approach used?"
    ]

    for q in queries:
        agent.query(q)