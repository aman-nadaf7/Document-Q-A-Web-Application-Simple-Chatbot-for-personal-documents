import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

document_store = {}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  
            chunks.append(chunk)
        start += chunk_size - overlap  

    return chunks


def embed_and_store(text: str, filename: str) -> tuple[str, int]:
    """
    Chunk text, embed each chunk, store in FAISS index.
    Returns (document_id, total_chunks).
    """
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Document produced no chunks after processing.")

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    doc_id = str(uuid.uuid4())
    document_store[doc_id] = {
        "chunks": chunks,
        "index": index,
        "filename": filename,
    }

    return doc_id, len(chunks)


def retrieve_relevant_chunks(document_id: str, question: str, top_k: int = 3) -> tuple[list[str], list[int]]:
    """
    Embed the question and find the top_k most relevant chunks.
    Returns (list_of_chunk_texts, list_of_chunk_indices_1_based).
    """
    if document_id not in document_store:
        raise KeyError(f"Document ID '{document_id}' not found.")

    store = document_store[document_id]
    chunks = store["chunks"]
    index = store["index"]

    query_embedding = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    sorted_indices = sorted(indices[0].tolist())
    selected_chunks = [chunks[i] for i in sorted_indices]
    source_numbers = [i + 1 for i in sorted_indices]

    return selected_chunks, source_numbers


def document_exists(document_id: str) -> bool:
    return document_id in document_store