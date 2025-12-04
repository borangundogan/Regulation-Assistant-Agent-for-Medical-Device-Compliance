# src/__main__.py
from .data_loader import load_regulation_text
from .chunking import chunk_text
from .hybrid_retriever import HybridRetriever


def main() -> None:
    print("Loading regulation text...")
    text = load_regulation_text()

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    print("Building hybrid retriever (this may take a moment)...")
    retriever = HybridRetriever(chunks)

    query = "What are the documentation requirements for a medical device?"
    print(f"\nQuery: {query}\n")

    results = retriever.retrieve(query, top_k=5)

    for i, (chunk, score) in enumerate(results, start=1):
        print(f"--- Result {i} (score={score:.3f}, id={chunk['id']}) ---")
        print(chunk["text"][:500])
        print()

    print("Done.")


if __name__ == "__main__":
    main()
