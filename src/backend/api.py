# src/medcompliance_agent/api.py

from fastapi import FastAPI
from pydantic import BaseModel

from medcompliance_agent.data_loader import load_regulation_text
from medcompliance_agent.chunking import chunk_text
from medcompliance_agent.hybrid_retriever import HybridRetriever
from medcompliance_agent.agent_graph import build_agent_graph

app = FastAPI(title="MedCompliance Agent API")


class DeviceRequest(BaseModel):
    device_name: str
    description: str | None = None


# Build shared components once at startup
text = load_regulation_text()
chunks = chunk_text(text)
retriever = HybridRetriever(chunks)
run_agent = build_agent_graph(retriever)


@app.post("/generate")
def generate_checklist(req: DeviceRequest):
    """
    Run the agent for a given device and return checklist + evaluation.
    """

    initial_state = {
        "device_info": {
            "name": req.device_name,
            "description": req.description or "",
        },
        # Optional: directly provide query instead of asking input()
        "query": f"documentation and regulatory requirements for {req.device_name}",
    }

    final_state = run_agent(initial_state)

    return {
        "device": final_state.get("device_info", {}),
        "checklist": final_state.get("checklist"),
        "evaluation": final_state.get("evaluation_json"),
    }
