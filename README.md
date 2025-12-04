# MedCompliance Agent (Prototype)

A small, educational project to simulate part of a **medical device compliance assistant**:

- Uses **RAG** (Retrieval-Augmented Generation) over regulatory text.
- Implements **hybrid retrieval** (dense vectors + BM25).
- Provides simple **memory abstractions** (short-term and long-term).
- Later, it can be extended with **LangGraph** agents and evaluation.

---

## 1. Motivation

This project is designed to practice and demonstrate skills that are relevant for:

- LLM-based agents
- RAG systems
- Hybrid retrieval
- Memory management for AI agents

It is inspired by use cases like **Formly.ai**, which help MedTech companies automate regulatory workflows.

---

## 2. Project Overview

Pipeline:

1. Load sample regulatory text from `data/regulations/`.
2. Split text into chunks.
3. Build:
   - BM25 index (sparse retrieval)
   - Dense vector index (embeddings)
4. Implement a **HybridRetriever** that:
   - Queries both indices
   - Combines scores
   - Returns top-k relevant chunks

Later extensions (optional):

- Add a simple **LangGraph agent** that:
  - Asks for device info
  - Queries regulations via the retriever
  - Generates a **first-draft compliance checklist**

---

## 3. Installation

```bash
git clone <this-repo-url>
cd medcompliance-agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
