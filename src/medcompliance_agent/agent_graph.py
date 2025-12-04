# src/medcompliance_agent/agent_graph.py

from typing import TypedDict, Dict, Any, List, Tuple
from langgraph.graph import StateGraph, END

from .hybrid_retriever import HybridRetriever

from .memory import ShortTermMemory, LongTermMemory, CheckpointStore
from pathlib import Path
from .config import PROJECT_ROOT

from .llm_client import call_llama

from .reasoning import analyze_requirements, make_checklist, refine_checklist

from .evaluation import evaluate_checklist

class AgentState(TypedDict, total=False):
    """
    Shared state for the agent.

    We keep this intentionally small and simple.
    """
    device_info: Dict[str, str]
    query: str
    retrieved: List[Tuple[Dict[str, Any], float]]
    checklist: str
    evaluation_human: str
    evaluation_json: Dict[str, Any]

def collect_device_info(state: AgentState) -> AgentState:
    """
    Simple CLI-based device info collection.

    In a real system this would come from a form or API.
    """
    if "device_info" in state:
        # Already provided (e.g., from API)
        return state

    print("=== Device Information ===")
    name = input("Device name (e.g., 'pulse oximeter'): ").strip()
    desc = input("Short description of the device: ").strip()

    device_info = {
        "name": name or "unknown device",
        "description": desc or "no description",
    }

    query = f"documentation and regulatory requirements for {device_info['name']}"

    new_state: AgentState = {
        **state,
        "device_info": device_info,
        "query": query,
    }
    return new_state


def retrieve_regulations(retriever: HybridRetriever):
    """
    Factory that returns a node function bound to our retriever.

    This lets us inject the retriever when we build the graph.
    """

    def _node(state: AgentState) -> AgentState:
        query = state.get("query")
        if not query:
            # Fallback: build a generic query
            device = state.get("device_info", {}).get("name", "a medical device")
            query_local = f"documentation requirements for {device}"
        else:
            query_local = query

        print(f"\n[Agent] Running hybrid retrieval for query: '{query_local}'")
        results = retriever.retrieve(query_local, top_k=5)

        new_state: AgentState = {
            **state,
            "retrieved": results,
        }
        return new_state

    return _node


def generate_checklist(state: AgentState) -> AgentState:
    retrieved = state.get("retrieved", [])
    device = state.get("device_info", {}).get("name", "the device")

    # Combine retrieved chunks
    context_text = "\n\n".join(chunk["text"] for chunk, _ in retrieved)

    # Step 1 — Analyze requirements
    analysis = analyze_requirements(device, context_text)

    # Step 2 — Extract present requirements only
    present = analysis  # LLM analysis already separates sections

    # Step 3 — Build initial checklist
    checklist_v1 = make_checklist(device, present)

    # Step 4 — Refine checklist
    checklist_v2 = refine_checklist(device, checklist_v1, context_text)

    new_state: AgentState = {
        **state,
        "checklist": checklist_v2,
    }
    return new_state

def evaluate_output(state: AgentState) -> AgentState:
    checklist = state.get("checklist", "")
    retrieved = state.get("retrieved", [])

    human, json_metrics = evaluate_checklist(checklist, retrieved)

    return {
        **state,
        "evaluation_human": human,
        "evaluation_json": json_metrics,
    }

def build_agent_graph(retriever: HybridRetriever):
    """
    Build and compile a simple LangGraph agent over the HybridRetriever.
    """
    # Memory paths
    ltm_path = PROJECT_ROOT / "data" / "memory" / "long_term.json"
    ckpt_path = PROJECT_ROOT / "data" / "memory" / "checkpoint.json"

    # Initialize memory systems
    short_term = ShortTermMemory()
    long_term = LongTermMemory(ltm_path)
    checkpoint = CheckpointStore(ckpt_path)

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("collect_device_info", collect_device_info)
    graph.add_node("retrieve_regulations", retrieve_regulations(retriever))
    graph.add_node("generate_checklist", generate_checklist)
    graph.add_node("evaluate_output", evaluate_output)

    # Edges (linear flow for now)
    graph.set_entry_point("collect_device_info")
    graph.add_edge("collect_device_info", "retrieve_regulations")
    graph.add_edge("retrieve_regulations", "generate_checklist")
    graph.add_edge("generate_checklist", "evaluate_output")
    graph.add_edge("evaluate_output", END)

    app = graph.compile()

    def run_with_memory(initial_state):
        # Try load checkpoint
        saved = checkpoint.load()
        if saved:
            print("\n[Agent] Resuming from checkpoint...")
            initial_state = saved

        # Run graph
        final_state = app.invoke(initial_state)

        # Save final checkpoint
        checkpoint.save(final_state)

        # Save to long-term memory
        if final_state.get("device_info"):
            long_term.add_record({
                "device": final_state["device_info"],
                "checklist": final_state.get("checklist"),
            })

        return final_state

    return run_with_memory    
