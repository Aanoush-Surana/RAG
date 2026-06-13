from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import uvicorn
from rag_engine import CodeRAGEngine

app = FastAPI(title="Graph-RAG Visualizer Backend")

# Enable CORS for local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared RAG instance
rag_engine = CodeRAGEngine()

class IngestRequest(BaseModel):
    repo_path: str
    force_reindex: bool = False

class QueryRequest(BaseModel):
    query: str
    api_key: str | None = None
    k: int = 4
    max_hops: int = 3
    max_expansion: int = 6
    include_callers: bool = False

class FileRequest(BaseModel):
    file_path: str


@app.get("/", response_class=HTMLResponse)
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "app.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="app.html not found")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/ingest")
def ingest_repository(req: IngestRequest):
    if not os.path.exists(req.repo_path):
        raise HTTPException(status_code=400, detail="Repository path does not exist.")
    try:
        rag_engine.load_repo(req.repo_path, force_reindex=req.force_reindex)
        return {
            "status": "success",
            "message": f"Successfully ingested {len(rag_engine.documents)} chunks.",
            "chunk_count": len(rag_engine.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def run_query(req: QueryRequest):
    if not rag_engine.documents:
        raise HTTPException(status_code=400, detail="No repository ingested yet.")
    try:
        # 1. Retrieve context chunks (including BFS hop information)
        retrieved = rag_engine.retrieve(
            req.query,
            k=req.k,
            max_hops=req.max_hops,
            max_expansion=req.max_expansion,
            include_callers=req.include_callers
        )
        
        # 2. Build context string
        context = rag_engine.build_context_string(retrieved)
        
        # 3. Generate answer
        answer = rag_engine.generate_answer(context, req.query, api_key=req.api_key)
        
        # 4. Format citation nodes for the UI
        citations = []
        for _, meta, hop in retrieved:
            citations.append({
                "name": meta["name"],
                "qname": meta["qname"],
                "type": meta["type"],
                "file": os.path.basename(meta["file"]),
                "full_path": meta["file"],
                "hop": hop
            })
            
        return {
            "answer": answer,
            "citations": citations
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
def run_query_stream(req: QueryRequest):
    if not rag_engine.documents:
        raise HTTPException(status_code=400, detail="No repository ingested yet.")
    try:
        # 1. Retrieve context chunks (including BFS hop information)
        retrieved = rag_engine.retrieve(
            req.query,
            k=req.k,
            max_hops=req.max_hops,
            max_expansion=req.max_expansion,
            include_callers=req.include_callers
        )
        
        # 2. Build context string
        context = rag_engine.build_context_string(retrieved)
        
        # 3. Format citations for the client
        citations = []
        for _, meta, hop in retrieved:
            citations.append({
                "name": meta["name"],
                "qname": meta["qname"],
                "type": meta["type"],
                "file": os.path.basename(meta["file"]),
                "full_path": meta["file"],
                "hop": hop
            })
            
        def event_generator():
            import json
            # First send the citations
            yield json.dumps({"type": "citations", "data": citations}) + "\n"
            
            # Then stream the text chunks
            for text_chunk in rag_engine.generate_answer_stream(context, req.query, api_key=req.api_key):
                yield json.dumps({"type": "content", "data": text_chunk}) + "\n"
                
        return StreamingResponse(event_generator(), media_type="application/x-ndjson")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph")
def get_graph():
    """Builds and returns graph nodes/edges for the D3 frontend."""
    if not rag_engine.documents:
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []
    
    # 1. Map name -> indices for resolving links
    name_to_ids = {}
    for i, c in enumerate(rag_engine.documents):
        name_to_ids.setdefault(c["name"], []).append(i)

    # 2. Build nodes list
    for i, c in enumerate(rag_engine.documents):
        nodes.append({
            "id": i,
            "name": c["name"],
            "qname": c["qname"],
            "type": c["type"],
            "file": os.path.basename(c["file"]),
            "full_path": c["file"],
            "doc": c.get("docstring", "")[:150],
            "methods": c.get("methods", []),
            "bases": c.get("base_classes", []),
        })

    # 3. Map method names -> class indices for obj.method() calls
    method_to_class_ids = {}
    for i, c in enumerate(rag_engine.documents):
        if c["type"] == "class":
            for method in c.get("methods", []):
                method_to_class_ids.setdefault(method, []).append(i)

    edge_set = set()
    def add_edge(src, tgt, kind):
        key = (src, tgt, kind)
        if key not in edge_set:
            edge_set.add(key)
            edges.append({"source": src, "target": tgt, "kind": kind})

    # 4. Populate calls and inheritance edges
    for i, c in enumerate(rag_engine.documents):
        caller_file = c["file"]
        
        # Resolve calls edges (prefer same-file matching first)
        for callee in c.get("calls", []):
            candidates = name_to_ids.get(callee, [])
            same_file = [j for j in candidates if rag_engine.documents[j]["file"] == caller_file and j != i]
            targets = same_file if same_file else [j for j in candidates if j != i]

            if not targets:
                owners = method_to_class_ids.get(callee, [])
                same_file2 = [j for j in owners if rag_engine.documents[j]["file"] == caller_file and j != i]
                targets = same_file2 if same_file2 else [j for j in owners if j != i]

            for j in targets:
                add_edge(i, j, "calls")

        # Resolve inheritance edges
        for base in c.get("base_classes", []):
            for j in name_to_ids.get(base, []):
                if j != i:
                    add_edge(i, j, "inherits")

    return {"nodes": nodes, "edges": edges}


@app.post("/api/file")
def get_file_content(req: FileRequest):
    """Returns the full code of a requested file."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        with open(req.file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
