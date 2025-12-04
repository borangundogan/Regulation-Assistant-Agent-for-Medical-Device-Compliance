# 🩺 MedCompliance Agent

Evidence-bound compliance assistant for medical devices.

This project is a fully working prototype of a regulatory AI agent that can:

- Generate compliance checklists for medical devices
- Use static MDR regulatory text or user-uploaded PDF regulations
- Perform hybrid retrieval (Dense + BM25)
- Run a LangGraph-based agent pipeline
- Evaluate output for hallucinations, redundancy, and format correctness
- Provide both a FastAPI backend and a Streamlit UI

It demonstrates real-world skills needed for agentic AI systems used in MedTech and enterprise automation.

## 1. 🔧 Features

✅ Hybrid Retrieval (BM25 + Dense Embeddings)

- BM25 for precise term matching
- Dense embeddings (nomic-embed-text) for semantic recall
- Score fusion → top-k combined results

✅ Dynamic PDF Ingestion

- Users can upload any regulation PDF, and the system:
  - Extracts text
  - Chunks it
  - Builds a temporary retriever
  - Runs the AI agent on only that regulation set

This makes it realistic for MedTech companies who work with MDR, IVDR, FDA guidance, IEC standards, etc.

✅ LangGraph Agent Pipeline

- The agent performs:
  - Device info collection
  - Hybrid retrieval
  - LLM-based reasoning
  - Checklist generation
  - Refinement
  - Output safety check
  - Evaluation (hallucination, redundancy, format validity)
  - Includes short-term memory, long-term memory, and checkpoint recovery.

✅ Evaluation Module

- Every output is automatically analyzed for:
  - Hallucinations
  - Missing items
  - Redundant checklist entries
  - Format validity
  - Basic coverage score

This is essential for safety-critical AI systems.

✅ Full UI + API

- Streamlit frontend
- FastAPI backend
- Both integrate with the agent pipeline

## 2. 🚀 Project Structure

```
src/
 ├── backend/
 │    └── api.py              # FastAPI backend (PDF upload + default mode)
 ├── frontend/
 │    └── ui.py               # Streamlit UI
 ├── medcompliance_agent/
 │    ├── agent_graph.py      # LangGraph agent
 │    ├── chunking.py         # Regulation text chunker
 │    ├── hybrid_retriever.py # Dense + BM25 hybrid search
 │    ├── evaluation.py       # Output evaluation
 │    ├── data_loader.py      # Load static regulation text
 │    ├── llm_client.py       # Local LLM (Ollama) calls
 │    ├── memory.py           # Short-term, long-term, checkpoint memory
 │    ├── pdf_utils.py        # PDF text extraction
 │    ├── reasoning.py        # Requirement analysis & checklist generation
 │    └── main.py             # CLI version
 ├── data/
 │    └── regulations/        # Static MDR-like example text
 ├── README.md
 ├── RULES.md
 └── pyproject.toml
```

## 3. 📦 Installation

Clone & enter project:

```bash
git clone <repo-url>
cd medcompliance-agent

Create environment with uv:
uv sync


(Or classic pip:)

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. ▶️ Running the System

Start backend (FastAPI):

```bash
uv run uvicorn backend.api:app --reload
```

Backend runs at:

http://127.0.0.1:8000

Start frontend (Streamlit UI):

```bash
uv run streamlit run src/frontend/ui.py
```

UI runs at:

http://localhost:8501

## 5. 📄 How It Works

Step 1 — Input

- User provides:
  - Device name
  - Short description
  - (Optional) A PDF file containing regulation text

Step 2 — Retrieval

- The system builds:
  - BM25 index
  - Dense embeddings
  - Hybrid retriever (score fusion)

Step 3 — Agent Execution (LangGraph)

- The agent:
  - Generates the search query
  - Retrieves relevant regulation chunks
  - Uses local LLM reasoning to produce a checklist
  - Refines and cleans the output
  - Validates safety (fallback if needed)
  - Evaluates the final checklist

Step 4 — Output

- UI displays:
  - Final checklist
  - Evaluation summary
  - Raw JSON evaluation

## 6. 🔬 Example Output

Checklist for pulse oximeter:

- Medical devices must include a technical documentation file
- The documentation shall include risk management, safety requirements, and performance evaluation
- Manufacturers must provide clinical data and post-market surveillance plans

Evaluation

- Hallucinations: 0
- Redundancy: 0.00
- Format: Valid
- Coverage: N/A
