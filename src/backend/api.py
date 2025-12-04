# src/backend/api.py

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from medcompliance_agent.data_loader import load_regulation_text
from medcompliance_agent.chunking import chunk_text
from medcompliance_agent.hybrid_retriever import HybridRetriever
from medcompliance_agent.agent_graph import build_agent_graph
from medcompliance_agent.pdf_utils import extract_text_from_pdf_bytes


app = FastAPI(title="MedCompliance Agent API")


class DeviceRequest(BaseModel):
    device_name: str
    description: str | None = None


# Static global pipeline
text = load_regulation_text()
chunks = chunk_text(text)
retriever = HybridRetriever(chunks)
run_agent = build_agent_graph(retriever)


@app.post("/generate")
def generate_checklist(req: DeviceRequest):
    """
    Old endpoint: use static regulation text bundled with the project.
    """
    initial_state = {
        "device_info": {
            "name": req.device_name,
            "description": req.description or "",
        },
        "query": f"documentation and regulatory requirements for {req.device_name}",
    }

    final_state = run_agent(initial_state)

    return {
        "device": final_state.get("device_info", {}),
        "checklist": final_state.get("checklist"),
        "evaluation": final_state.get("evaluation_json"),
    }


# dynamic endpoint: user uploaded PDF 

@app.post("/generate_from_pdf")
async def generate_from_pdf(
    pdf: UploadFile = File(...),
    device_name: str = Form(...),
    description: str = Form(""),
):
    """
    New endpoint: user uploads a regulation PDF.
    We extract text, chunk it, build a temporary retriever+agent,
    and run the pipeline just for this request.
    """
    pdf_bytes = await pdf.read()

    raw_text = extract_text_from_pdf_bytes(pdf_bytes)

    chunks = chunk_text(raw_text)

    local_retriever = HybridRetriever(chunks)
    local_run_agent = build_agent_graph(local_retriever)

    initial_state = {
        "device_info": {
            "name": device_name,
            "description": description,
        },
        "query": f"documentation and regulatory requirements for {device_name}",
    }

    final_state = local_run_agent(initial_state)

    return {
        "device": final_state.get("device_info", {}),
        "checklist": final_state.get("checklist"),
        "evaluation": final_state.get("evaluation_json"),
    }
