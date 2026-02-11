
# ======================== FILE: main.py ========================
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uuid, shutil, json

from config import settings
from orchestrator import Orchestrator
from processors import PDFProcessor, ImageProcessor, CSVProcessor
from vectorstore import store_manager
from bm25_store import BM25Store
from graph_store import KnowledgeGraph

app = FastAPI(title="Multi-Agent RAG System", version="1.0")
orchestrator = Orchestrator()


@app.get("/")
async def root():
    return {"message": "Multi-Agent RAG API is running 🚀"}

# Processors
processors = {
    "pdf": PDFProcessor(),
    "image": ImageProcessor(),
    "csv": CSVProcessor(),
}
ext_map = {
    ".pdf": "pdf",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".csv": "csv", ".xlsx": "csv", ".xls": "csv",
}

# Shared stores
bm25 = BM25Store("hybrid")
kg = KnowledgeGraph()
# Inject into agents that need them
orchestrator.agents["hybrid_rag"].bm25 = bm25
orchestrator.agents["graph_rag"].kg = kg


# ---- UPLOAD ----
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    file_id = str(uuid.uuid4())
    ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    file_type = ext_map.get(ext, "pdf")

    # Save file
    file_path = str(settings.upload_dir / f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    meta = json.loads(metadata) if metadata else {}
    meta.update({"file_id": file_id, "source_filename": file.filename})

    # Process
    processor = processors[file_type]
    result = processor.process(file_path, meta)

    # Index into ALL relevant stores so every agent can use the data
    counts = {}

    # Naive RAG, Agentic RAG, HyDE RAG, Corrective RAG, Hybrid RAG all share this
    n = store_manager.add_documents("naive_rag", result["standard_chunks"])
    counts["naive_rag"] = n

    # Sentence Window RAG
    n = store_manager.add_documents("sentence_window_rag", result["sentence_chunks"])
    counts["sentence_window_rag"] = n

    # Parent-Child RAG
    store_manager.add_documents("parent_child_rag_children", result["child_chunks"])
    n = store_manager.add_documents("parent_child_rag_parents", result["parent_chunks"])
    counts["parent_child_rag"] = n

    # Multimodal RAG (always index — text docs are searchable too)
    n = store_manager.add_documents("multimodal_rag", result["standard_chunks"])
    counts["multimodal_rag"] = n

    # Table RAG (only for structured data, but schema is useful)
    if file_type == "csv":
        n = store_manager.add_documents("table_rag", result["standard_chunks"])
        counts["table_rag"] = n

    # BM25 for Hybrid RAG
    bm25.add_documents(result["standard_chunks"])
    counts["bm25"] = len(result["standard_chunks"])

    # Knowledge Graph for Graph RAG
    if result["full_text"]:
        kg.extract_and_add(result["full_text"][:5000], source=file.filename)
        counts["graph_rag"] = kg.get_stats()

    return JSONResponse(content={
        "file_id": file_id,
        "filename": file.filename,
        "file_type": file_type,
        "index_counts": counts,
        "status": "indexed_to_all_agents",
    })


# ---- QUERY (auto-routed) ----
@app.post("/query")
async def query(
    question: str = Form(...),
    top_k: int = Form(5),
):
    result = await orchestrator.query(question, top_k=top_k)
    return JSONResponse(content=result)


# ---- QUERY SPECIFIC AGENT ----
@app.post("/query/{agent_name}")
async def query_agent(
    agent_name: str,
    question: str = Form(...),
    top_k: int = Form(5),
):
    result = await orchestrator.query(question, agent_name=agent_name, top_k=top_k)
    return JSONResponse(content=result)


# ---- COMPARE AGENTS ----
@app.post("/compare")
async def compare(
    question: str = Form(...),
    agents: str = Form("naive_rag,agentic_rag,hybrid_rag"),
    top_k: int = Form(5),
):
    agent_list = [a.strip() for a in agents.split(",")]
    result = await orchestrator.compare(question, agent_list, top_k)
    return JSONResponse(content=result)


# ---- LIST AGENTS ----
@app.get("/agents")
async def list_agents():
    return orchestrator.list_agents()


# ---- COLLECTION STATS ----
@app.get("/collections")
async def list_collections():
    return {
        "vector_stores": store_manager.list_collections(),
        "knowledge_graph": kg.get_stats(),
    }


# ---- HEALTH ----
@app.get("/health")
async def health():
    return {"status": "running", "agents": len(orchestrator.agents)}

