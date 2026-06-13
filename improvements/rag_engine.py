import ast
import hashlib
import os
import re
import math
from collections import defaultdict, deque, Counter
import numpy as np

# Skip common noise directories
SKIP_DIRS = {
    "venv", ".venv", "env", ".env",
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", "site-packages",
    ".tox", "eggs", ".eggs",
}

def _get_docstring(node):
    """Extract the first docstring from a function or class node, if any."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value.strip()[:300]
    return ""


def _get_calls(node):
    """Return the sorted set of function/method names called inside `node`."""
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
    """Return a list of decorator names on a function/class node."""
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
    """Return base-class names for an ast.ClassDef node."""
    bases = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(base.attr)
    return bases


def _get_methods(class_node):
    """Return method names of a class."""
    return [
        item.name
        for item in class_node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


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


def parse_python_file(file_path, repo_path=None):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except Exception as e:
        print(f"[ERROR] Cannot open {file_path}: {e}")
        return []

    if len(code) > 80_000:
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
        if not segment or len(segment) < 10:
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


def dedup_chunks(chunks):
    seen   = {}
    unique = []

    for chunk in chunks:
        key = (chunk["name"], chunk["content_hash"])
        if key in seen:
            unique[seen[key]]["also_in"].append(os.path.basename(chunk["file"]))
        else:
            seen[key] = len(unique)
            unique.append(chunk)
    return unique


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


def tokenize_code(code_text):
    """Tokenize code text, splitting camelCase and snake_case and keeping keywords."""
    words = re.findall(r'[a-zA-Z0-9]+', code_text.lower())
    split_words = []
    for word in words:
        # Split camelCase
        subwords = re.sub('([a-z0-9])([A-Z])', r'\1 \2', word).split()
        split_words.extend(subwords)
    return [w.lower() for w in split_words if len(w) > 1]


class BM25:
    """A pure-Python implementation of BM25 retrieval for code chunks."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(docs)
        self.doc_freqs = []
        self.doc_lengths = []
        self.df = Counter()
        
        for doc in docs:
            tokens = tokenize_code(doc)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lengths.append(len(tokens))
            for word in freqs:
                self.df[word] += 1
                
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0
        self.idf = {}
        for word, freq in self.df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens):
        scores = [0.0] * self.corpus_size
        for word in query_tokens:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            for i in range(self.corpus_size):
                freq = self.doc_freqs[i].get(word, 0)
                doc_len = self.doc_lengths[i]
                denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


class CodeRAGEngine:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.metadata = []
        self.index = None
        self.embed_model = None
        self.bm25 = None
        self.symbol_index = {}
        self.local_tokenizer = None
        self.local_llm = None
        self._name_index = defaultdict(list)
        self.repo_path = ""

    def load_repo(self, repo_path, force_reindex=False):
        self.repo_path = repo_path
        
        cache_dir = os.path.join(repo_path, ".graph_rag_cache")
        metadata_file = os.path.join(cache_dir, "metadata.json")
        index_file = os.path.join(cache_dir, "faiss.index")
        
        import faiss
        import json
        from sentence_transformers import SentenceTransformer
        
        # Check if cache exists and we are not forcing re-indexing
        if not force_reindex and os.path.exists(metadata_file) and os.path.exists(index_file):
            print(f"[RAG Engine] Loading cache from {cache_dir}...")
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                
                self.index = faiss.read_index(index_file)
                self.chunks = [d["content"] for d in self.documents]
                self.metadata = self.documents
                
                # Rebuild symbol index & name index
                self.symbol_index = {}
                self._name_index = defaultdict(list)
                for i, doc in enumerate(self.documents):
                    self._name_index[doc["name"]].append(i)
                    self.symbol_index[doc["name"].lower()] = i
                    if doc["type"] == "class":
                        for method in doc.get("methods", []):
                            self.symbol_index[method.lower()] = i
                            
                self.bm25 = BM25(self.chunks)
                self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                
                print(f"[RAG Engine] Loaded {len(self.documents)} chunks from persistent cache successfully.")
                return
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}. Rebuilding index...")
                
        # If cache not found or force_reindex is True, compute it:
        self.documents = []
        file_count = 0
        
        # Load and parse python files
        for root, dirs, files in os.walk(repo_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(root, file)
                self.documents.extend(parse_python_file(file_path, repo_path))
                file_count += 1

        self.documents = dedup_chunks(self.documents)
        self.documents = [d for d in self.documents if len(d["content"].strip()) > 10]
        
        if not self.documents:
            raise ValueError("No valid Python chunks found.")
            
        build_reverse_index(self.documents)
        
        # Build symbol index & name index
        self.symbol_index = {}
        self._name_index = defaultdict(list)
        for i, doc in enumerate(self.documents):
            self._name_index[doc["name"]].append(i)
            self.symbol_index[doc["name"].lower()] = i
            if doc["type"] == "class":
                for method in doc.get("methods", []):
                    self.symbol_index[method.lower()] = i

        self.chunks = [d["content"] for d in self.documents]
        self.metadata = self.documents
        
        self.bm25 = BM25(self.chunks)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        embed_texts = [
            f"{d['name']}: {d['docstring']}\n{d['content']}"
            for d in self.documents
        ]
        embeddings = self.embed_model.encode(embed_texts, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))
        
        # Save cache
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            faiss.write_index(self.index, index_file)
            print(f"[RAG Engine] Saved persistent cache to {cache_dir}.")
        except Exception as e:
            print(f"[WARNING] Failed to write cache: {e}")
            
        print(f"[RAG Engine] Successfully indexed {file_count} files ({len(self.documents)} chunks).")

    def init_local_llm(self):
        """Lazy load local LLM if needed."""
        if self.local_llm is not None:
            return
            
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("[RAG Engine] Loading Qwen2.5-Coder-1.5B local LLM...")
        LLM_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        self.local_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
        self.local_llm = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            torch_dtype=torch.float32,
        )
        self.local_llm.eval()
        print("[RAG Engine] Local LLM ready.")

    def hybrid_retrieve(self, query, k=5):
        """Perform Hybrid Search: Vector + BM25 + Exact Symbol Boost."""
        query_tokens = tokenize_code(query)
        
        # 1. Dense Vector Search
        q_emb = self.embed_model.encode([query])
        distances, vector_indices = self.index.search(np.array(q_emb, dtype="float32"), len(self.chunks))
        vector_rankings = {idx: rank for rank, idx in enumerate(vector_indices[0])}
        
        # 2. Sparse BM25 Search
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1]
        bm25_rankings = {idx: rank for rank, idx in enumerate(bm25_indices)}
        
        # 3. Exact Symbol Lookup Boost
        symbol_boost = set()
        for token in query_tokens:
            if len(token) > 3 and token in self.symbol_index:
                symbol_boost.add(self.symbol_index[token])
                
        # 4. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for idx in range(len(self.chunks)):
            v_rank = vector_rankings.get(idx, 9999)
            b_rank = bm25_rankings.get(idx, 9999)
            
            # RRF Formula: 1 / (60 + rank)
            score = 1.0 / (60.0 + v_rank) + 1.0 / (60.0 + b_rank)
            
            # Massive boost if query matches class/method signature exactly
            if idx in symbol_boost:
                score += 1.5
                
            rrf_scores[idx] = score
            
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for idx in sorted_indices[:k]:
            results.append((self.chunks[idx], self.metadata[idx], 0))  # hop=0
            
        return results, sorted_indices[:k]

    def _expand_with_graph_multihop(self, retrieved_indices, max_hops=3, max_expansion=8, include_callers=False):
        if max_hops == 0 or not retrieved_indices:
            return []
            
        frontier = deque()
        seen = set(retrieved_indices)
        expanded = []
        
        for idx in retrieved_indices:
            frontier.append((idx, 0))
            
        while frontier and len(expanded) < max_expansion:
            idx, hop = frontier.popleft()
            if hop >= max_hops:
                continue
                
            meta = self.metadata[idx]
            neighbors = list(meta["base_classes"]) + list(meta["calls"])
            if include_callers:
                neighbors += list(meta.get("called_by", []))
                
            for name in neighbors:
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
                    
                    if len(expanded) < max_expansion:
                        frontier.append((neighbor_idx, next_hop))
                    else:
                        break
                if len(expanded) >= max_expansion:
                    break
        return expanded

    def retrieve(self, query, k=4, max_hops=3, max_expansion=6, include_callers=False):
        # Hybrid retrieval (hop-0)
        direct_results, direct_indices = self.hybrid_retrieve(query, k=k)
        
        # Graph BFS traversal (hop >= 1)
        expanded = []
        if max_hops > 0:
            expanded = self._expand_with_graph_multihop(
                direct_indices,
                max_hops=max_hops,
                max_expansion=max_expansion,
                include_callers=include_callers
            )
            
        return direct_results + expanded

    def generate_answer(self, context, query, api_key=None):
        system_prompt = (
            "You are a senior software engineer. "
            "Answer questions about the codebase using ONLY the provided code context. "
            "Refer to structural relationships (extends, calls, called_by, methods) to trace functionality. "
            "Keep code complete, accurate, and explain how classes or functions work. "
            "Cite the file names and symbols when referencing them (e.g. `[main.py::SegmentationModel]`)."
        )
        
        user_prompt = f"Code context:\n{context}\n\nQuestion: {query}"
        
        if api_key:
            # Call Gemini API via requests (standard REST endpoint)
            import requests
            import json
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                }
            }
            try:
                r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=30)
                r.raise_for_status()
                res = r.json()
                return res["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                print(f"[ERROR] Gemini API request failed: {e}. Falling back to local LLM.")
                # Fall back to local model if API fails
        
        # Local model execution
        self.init_local_llm()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        text = self.local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.local_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to("cpu")
        
        input_len = inputs["input_ids"].shape[1]
        
        import torch
        with torch.no_grad():
            output_ids = self.local_llm.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.local_tokenizer.eos_token_id,
            )
            
        new_tokens = output_ids[0][input_len:]
        return self.local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_answer_stream(self, context, query, api_key=None):
        system_prompt = (
            "You are a senior software engineer. "
            "Answer questions about the codebase using ONLY the provided code context. "
            "Refer to structural relationships (extends, calls, called_by, methods) to trace functionality. "
            "Keep code complete, accurate, and explain how classes or functions work. "
            "Cite the file names and symbols when referencing them (e.g. `[main.py::SegmentationModel]`)."
        )
        
        user_prompt = f"Code context:\n{context}\n\nQuestion: {query}"
        
        if api_key:
            import requests
            import json
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                }
            }
            try:
                r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), stream=True, timeout=30)
                r.raise_for_status()
                buffer = ""
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    clean_line = line.strip()
                    if clean_line.startswith("["):
                        clean_line = clean_line[1:].strip()
                    if clean_line.startswith(","):
                        clean_line = clean_line[1:].strip()
                    if clean_line.endswith("]"):
                        clean_line = clean_line[:-1].strip()
                    if not clean_line:
                        continue
                    buffer += clean_line
                    try:
                        data = json.loads(buffer)
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        yield text
                        buffer = ""
                    except json.JSONDecodeError:
                        continue
                return
            except Exception as e:
                print(f"[ERROR] Gemini Streaming failed: {e}. Falling back to local LLM.")
                # Fall back to local model if API fails
        
        # Local model execution
        self.init_local_llm()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        text = self.local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.local_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to("cpu")
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(self.local_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=self.local_tokenizer.eos_token_id,
        )
        thread = Thread(target=self.local_llm.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text

    def build_context_string(self, retrieved):
        """Construct prompt context using entire semantic code blocks."""
        context_parts = []
        for chunk, meta, hop in retrieved:
            hop_label = "direct" if hop == 0 else f"hop-{hop}"
            header = (
                f"# type: {meta['type']} | name: {meta['name']} | "
                f"file: {os.path.basename(meta['file'])} | retrieval: {hop_label}\n"
            )
            if meta.get("base_classes"):
                header += f"# extends  : {', '.join(meta['base_classes'])}\n"
            if meta.get("methods"):
                header += f"# methods  : {', '.join(meta['methods'][:8])}\n"
            if meta.get("calls"):
                header += f"# calls    : {', '.join(meta['calls'][:6])}\n"
                
            # Send the complete code block directly, no aggressive/truncating clean
            context_parts.append(header + chunk)
            
        return "\n\n---\n\n".join(context_parts)
