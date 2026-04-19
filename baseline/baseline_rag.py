# =========================
# IMPORTS
# =========================
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# LOAD REPO (NO ingest)
# =========================
def load_repo(repo_path):
    code_text = ""

    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".js", ".html", ".css", ".json", ".md", ".txt")):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        code_text += f"\n\n### FILE: {file_path} ###\n"
                        code_text += content
                except:
                    continue

    return code_text


# =========================
# RAG AGENT
# =========================
class RAGCodebaseAgent:

    def __init__(self, repo_path, chunk_size=1000, chunk_overlap=200):
        print(f"[RAG] Loading repo from: {repo_path}")

        raw_text = load_repo(repo_path)

        # ===== CHUNKING =====
        self.chunks = self.chunk_text(raw_text, chunk_size, chunk_overlap)
        print(f"[RAG] Total chunks: {len(self.chunks)}")

        # ===== EMBEDDINGS =====
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embed_model.encode(self.chunks)

        # ===== FAISS =====
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

        # ===== LOCAL LLM =====
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        print("[RAG] Ready.")

    # =========================
    # CHUNKING
    # =========================
    def chunk_text(self, text, chunk_size, overlap):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    # =========================
    # RETRIEVAL
    # =========================
    def retrieve(self, query, k=5):
        q_emb = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(q_emb), k)

        retrieved = [self.chunks[i] for i in indices[0]]

        print("\n--- RETRIEVED CHUNKS ---")
        for i, chunk in enumerate(retrieved):
            print(f"\nChunk {i+1}:\n{chunk[:300]}")

        return retrieved

    # =========================
    # LLM
    # =========================
    def run_llm(self, context, query):
        prompt = f"""
Use ONLY the given context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = self.llm.generate(**inputs, max_new_tokens=100)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # =========================
    # QUERY
    # =========================
    def query(self, question, k=5):
        print("\n==============================")
        print("QUERY:", question)

        docs = self.retrieve(question, k)

        context = "\n\n---\n\n".join(docs)

        print("\n--- FINAL CONTEXT ---")
        print(context[:500])

        answer = self.run_llm(context, question)

        print("\n--- MODEL OUTPUT ---")
        print(answer)

        return answer


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # CHANGE THIS TO YOUR FOLDER
    repo_path = r"C:\Users\Aanoush Surana\OneDrive\Desktop\BTP\BASIC_RAG"

    agent = RAGCodebaseAgent(repo_path)

    queries = [
        "What does this project do?",
        "How is data processed?",
        "Which file handles API requests?",
        "Explain the main architecture",
        "How does the system store data?",
        "Why is this approach used?"
    ]

    for q in queries:
        agent.query(q)