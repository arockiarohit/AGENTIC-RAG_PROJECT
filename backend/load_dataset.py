import os
import json
from app.vector_store import vector_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "theni_complete_large_rag_dataset.json")

print("=" * 50)
print("RAG DATASET LOADER STARTED")
print("=" * 50)


def load_dataset():
    """
    Load JSON dataset and store embeddings in FAISS.
    """
    vector_store.clear()

    print("Opening dataset...")

    # Read JSON file
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from dataset")

    chunks = []

    # Convert JSON records into vector-store-compatible chunks
    for item in data:
        chunks.append({
            "content": item.get("content", ""),
            "metadata": {
                "chunk_id": item.get("chunk_id"),
                "section": item.get("section"),
                "keywords": item.get("keywords", []),
                "source": "theni_complete_large_rag_dataset.json"
            }
        })

    print(f"Prepared {len(chunks)} chunks for embedding")

    # Add chunks to FAISS vector store
    vector_store.add_documents(chunks)

    print("=" * 50)
    print("DATASET EMBEDDED SUCCESSFULLY")
    print("=" * 50)

    # Print vector database statistics
    stats = vector_store.get_stats()

    print("\nFAISS VECTOR STORE STATS")
    print("-" * 50)
    print(f"Total Vectors   : {stats['total_vectors']}")
    print(f"Total Documents : {stats['total_documents']}")
    print(f"Embedding Size  : {stats['dimension']}")
    print("-" * 50)


if __name__ == "__main__":
    print("RUNNING MAIN")
    load_dataset()
