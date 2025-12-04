# src/backend/api.py

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from medcompliance_agent.data_loader import load_regulation_text
from medcompliance_agent.chunking import chunk_text, chunk_pdf_with_sections
from medcompliance_agent.hybrid_retriever import HybridRetriever
from medcompliance_agent.agent_graph import build_agent_graph
from medcompliance_agent.pdf_utils import extract_text_from_pdf_bytes

from medcompliance_agent.vector_store import QdrantStore

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="MedCompliance Agent API")


class DeviceRequest(BaseModel):
    device_name: str
    description: str | None = None


# dynamic endpoint: user uploaded PDF 
@app.post("/generate_from_pdf")
async def generate_from_pdf(
    pdf: UploadFile = File(...),
    device_name: str = Form(...),
    description: str = Form(""),
):
    """
    User-uploaded PDF:
    - extract the text
    - segment into sections
    - chunk into RAG pieces
    - embed + store in Qdrant
    - perform hybrid (dense + sparse) retrieval
    - run LangGraph agent on top
    """

    pdf_bytes = await pdf.read()
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)

    source_name = pdf.filename or "uploaded_pdf"
    chunks = chunk_pdf_with_sections(raw_text, source_name=source_name)

    qdrant = QdrantStore(collection_name="user_regulations")

    def embed_fn(text: str):
        return embedding_model.encode(text).tolist()

    qdrant.add_chunks(chunks, embed_fn)

    retriever = HybridRetriever(
        qdrant=qdrant,
        chunks=chunks,
        embed_fn=embed_fn
    )

    run_agent = build_agent_graph(retriever)

    initial_state = {
        "device_info": {
            "name": device_name,
            "description": description,
        },
        "query": f"documentation and regulatory requirements for {device_name}",
    }

    final_state = run_agent(initial_state)

    return {
        "device": final_state.get("device_info", {}),
        "checklist": final_state.get("checklist"),
        "evaluation": final_state.get("evaluation_json"),
    }
