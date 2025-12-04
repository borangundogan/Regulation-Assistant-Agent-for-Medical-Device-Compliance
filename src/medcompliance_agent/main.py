# src/medcompliance_agent/main.py

from .data_loader import load_regulation_text
from .chunking import chunk_text
from .hybrid_retriever import HybridRetriever
from .agent_graph import build_agent_graph


def main():
    print("Loading regulation text...")
    text = load_regulation_text()

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    print("Building hybrid retriever (this may take a moment)...")
    retriever = HybridRetriever(chunks)

    # Build LangGraph app
    print("Compiling LangGraph agent...")
    app = build_agent_graph(retriever)

    print("\n=== Starting MedCompliance Agent ===")
    print("The agent will ask you some questions about your device.\n")

    # Initial empty state
    initial_state = {}

    # Run the agent once (single pass)
    final_state = app.invoke(initial_state)

    checklist = final_state.get("checklist")
    device_info = final_state.get("device_info", {})

    print("\n=== Agent Output ===")
    print(f"Device: {device_info.get('name', 'unknown')}")
    print()
    print(checklist or "No checklist generated.")


if __name__ == "__main__":
    main()
