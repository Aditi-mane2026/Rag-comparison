from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv

# Load env variables (for OPENAI_API_KEY)
load_dotenv(".env")

import sys
import os
import asyncio

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, "shared"))
sys.path.append(os.path.join(base_dir, "traditional rag"))
sys.path.append(os.path.join(base_dir, "C-rag"))
sys.path.append(os.path.join(base_dir, "self_rag"))

from vector_store import build_vector_store
from strictrag import run_strict_rag
from openrag import run_open_rag
from crag import run_crag
from self_rag import run_self_rag

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Building vector store on startup from apple_fruit_confusion_doc_v2.pdf...")
    pdf_path = os.path.join(base_dir, "apple_fruit_confusion_doc_v2.pdf")
    if os.path.exists(pdf_path):
        app.state.retriever = build_vector_store(pdf_path)
        print("Vector store ready!")
    else:
        print(f"File not found: {pdf_path}")
        app.state.retriever = None
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global store for retriever to avoid rebuilding constantly in this demo app
app.state.retriever = None

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
def get_ui():
    with open(os.path.join(base_dir, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/query")
async def query_rag(request: QueryRequest):
    if app.state.retriever is None:
        raise HTTPException(status_code=400, detail="Please upload a document first to build the index.")
    
    query = request.query
    
    try:
        # Execute pipelines concurrently using asyncio threads to drastically reduce response times
        task_strict = asyncio.to_thread(run_strict_rag, app.state.retriever, query)
        task_open = asyncio.to_thread(run_open_rag, app.state.retriever, query)
        task_crag = asyncio.to_thread(run_crag, app.state.retriever, query)
        task_self = asyncio.to_thread(run_self_rag, app.state.retriever, query)
        
        # Wait for all to complete simultaneously
        res_strict, res_open, res_crag, res_self = await asyncio.gather(
            task_strict, task_open, task_crag, task_self
        )
        
        return {
            "strict_rag": res_strict,
            "open_rag": res_open,
            "crag": res_crag,
            "self_rag": res_self
        }
    except Exception as e:
        print(f"Error during querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
