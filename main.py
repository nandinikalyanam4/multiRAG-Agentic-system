
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uuid, shutil, json

from config import settings
from rl_orchestrator import RLOrchestrator
from processors import PDFProcessor, ImageProcessor, CSVProcessor
from vectorstore import store_manager
from bm25_store import BM25Store
from graph_store import KnowledgeGraph

app = FastAPI(title="RL-Enhanced Multi-Agent RAG System", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = RLOrchestrator()

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

bm25 = BM25Store("hybrid")
kg = KnowledgeGraph()
orchestrator.agents["hybrid_rag"].bm25 = bm25
orchestrator.agents["graph_rag"].kg = kg


@app.get("/")
async def root():
    return {"message": "RL-Enhanced Multi-Agent RAG API v2.0"}


# ---- UPLOAD (same as before) ----
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    file_id = str(uuid.uuid4())
    ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    file_type = ext_map.get(ext, "pdf")

    file_path = str(settings.upload_dir / f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        meta = json.loads(metadata) if metadata else {}
    except (json.JSONDecodeError, TypeError):
        meta = {}

    meta.update({"file_id": file_id, "source_filename": file.filename})

    processor = processors[file_type]
    result = processor.process(file_path, meta)

    counts = {}
    n = store_manager.add_documents("naive_rag", result["standard_chunks"])
    counts["naive_rag"] = n
    n = store_manager.add_documents("sentence_window_rag", result["sentence_chunks"])
    counts["sentence_window_rag"] = n
    store_manager.add_documents("parent_child_rag_children", result["child_chunks"])
    n = store_manager.add_documents("parent_child_rag_parents", result["parent_chunks"])
    counts["parent_child_rag"] = n
    n = store_manager.add_documents("multimodal_rag", result["standard_chunks"])
    counts["multimodal_rag"] = n
    if file_type == "csv":
        n = store_manager.add_documents("table_rag", result["standard_chunks"])
        counts["table_rag"] = n
    bm25.add_documents(result["standard_chunks"])
    counts["bm25"] = len(result["standard_chunks"])
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


# ---- QUERY (now uses bandit routing) ----
@app.post("/query")
async def query(
    question: str = Form(...),
    top_k: int = Form(5),
):
    result = await orchestrator.query(question, top_k=top_k)
    return JSONResponse(content=result)


@app.post("/query/{agent_name}")
async def query_agent(
    agent_name: str,
    question: str = Form(...),
    top_k: int = Form(5),
):
    result = await orchestrator.query(question, agent_name=agent_name, top_k=top_k)
    return JSONResponse(content=result)


# ---- COMPARE ----
@app.post("/compare")
async def compare(
    question: str = Form(...),
    agents: str = Form("naive_rag,agentic_rag,hybrid_rag"),
    top_k: int = Form(5),
):
    agent_list = [a.strip() for a in agents.split(",")]
    result = await orchestrator.compare(question, agent_list, top_k)
    return JSONResponse(content=result)


# ---- FEEDBACK (NEW — the reward signal) ----
@app.post("/feedback")
async def submit_feedback(
    interaction_id: str = Form(...),
    reward: float = Form(...),  # 0.0 = terrible, 1.0 = perfect
):
    if not 0.0 <= reward <= 1.0:
        raise HTTPException(400, "Reward must be between 0.0 and 1.0")
    result = orchestrator.submit_feedback(interaction_id, reward)
    return JSONResponse(content=result)


# ---- RL STATS (NEW — see what the bandit has learned) ----
@app.get("/rl/stats")
async def rl_stats():
    return {
        "bandit_state": orchestrator.bandit.get_stats(),
        "description": "Average reward per agent per question category. Higher = better. The bandit uses these to route queries.",
    }


# ---- RL LEADERBOARD (NEW — ranked agent performance) ----
@app.get("/rl/leaderboard")
async def rl_leaderboard():
    return {
        "leaderboard": orchestrator.bandit.get_leaderboard(),
        "description": "Agents ranked by overall average reward across all categories.",
    }


# ---- RL OPTIMIZE (NEW — run self-improvement analysis) ----
@app.post("/rl/optimize")
async def rl_optimize():
    report = orchestrator.improver.analyze()
    return JSONResponse(content=report)


# ---- RL RESET (NEW — reset bandit to start fresh) ----
@app.post("/rl/reset")
async def rl_reset():
    orchestrator.bandit.reset()
    return {"status": "bandit_reset_to_initial_priors"}


# ---- OTHER ENDPOINTS (same as before) ----
@app.get("/agents")
async def list_agents():
    return orchestrator.list_agents()


@app.get("/collections")
async def list_collections():
    return {
        "vector_stores": store_manager.list_collections(),
        "knowledge_graph": kg.get_stats(),
    }


@app.get("/health")
async def health():
    return {"status": "running", "agents": len(orchestrator.agents), "version": "2.0-rl"}