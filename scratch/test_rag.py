import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_store import vector_store
from app.config import OLLAMA_MODEL

def test():
    print(f"Ollama Model: {OLLAMA_MODEL}")
    question = "tell me about church"
    print(f"Question: {question}")
    
    results = vector_store.search(question)
    print("Retrieved chunks:")
    for res in results:
        print(f" - score: {res['score']}")
        print(f"   content: {res['content'][:150]}...")
        
    print("\nLLM Output:")
    from app.rag_chain import ask_question
    for chunk in ask_question(question):
        print(chunk, end="")
    print("\n---")

if __name__ == "__main__":
    test()
