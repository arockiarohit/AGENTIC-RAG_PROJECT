import asyncio
import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat(question):
    print(f"\n--- Testing Question: {question} ---")
    response = client.post("/chat", json={"question": question, "session_id": "test_session_1"})
    print("Status:", response.status_code)
    print("Response:", response.text)

if __name__ == "__main__":
    # Test 1 — Theni
    test_chat("What are the major tourist places in Theni?")
    
    # Test 2 — Theni-specific question without explicit "Theni"
    test_chat("Which waterfalls are popular in the district?")
    
    # Test 3 — General
    test_chat("What is LangGraph?")
    
    # Test 4 — Current information
    test_chat("What is the current population of Theni district?")
    
    # Test 5 — Completely unrelated
    test_chat("What is the capital of Japan?")
