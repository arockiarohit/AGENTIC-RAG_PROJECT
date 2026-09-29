"""
Text chunking module.
Splits document text into overlapping chunks for embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, filename: str = "unknown") -> list[dict]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        text: The full document text to chunk.
        filename: Source filename for metadata.
    
    Returns:
        List of dicts with 'content' and 'metadata' keys.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "content": chunk,
            "metadata": {
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        })

    return documents
