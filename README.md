# Document Q&A App

A simple web app I built where you upload a `.txt` file and ask questions about it. The important thing is it only answers from what's in the document — it won't make stuff up from general knowledge.

---

## What I used

| Part       | Technology |
|------------|------------|
| Backend    | FastAPI (Python) |
| LLM        | Groq — `llama-3.3-70b-versatile` |
| Embeddings | `all-MiniLM-L6-v2` from Sentence Transformers |
| Vector Search | FAISS (runs in memory) |
| Frontend   | Plain HTML, CSS, JS — no frameworks |

---

## Why these choices

I went with **Groq** because it's genuinely fast and the free tier is more than enough for this. The `llama-3.3-70b-versatile` model is good at sticking to instructions, which matters here since the whole point is to not answer outside the document.

For embeddings I used **all-MiniLM-L6-v2** — it runs locally, no API key needed, and it works really well for this kind of semantic search use case.

---

## Folder structure

```
document_qa/
├── backend/
│   ├── main.py           ← API routes
│   ├── rag.py            ← chunking, embeddings, retrieval
│   ├── models.py         ← request/response models
│   ├── requirements.txt
│ 
├── frontend/
│   └── index.html        ← the whole UI in one file
└── README.md
```

---

## How to run it

### 1. Get a Groq API key

Go to https://console.groq.com, sign up and create a free API key. Takes 2 minutes.

### 2. Add your API key

Open `backend/.env` and put your key in:

```
GROQ_API_KEY=your_key_here
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

First time will take a few minutes since it downloads the embedding model too.

### 4. Start the backend

```bash
uvicorn main:app --reload
```

It runs at `http://127.0.0.1:8000`. Keep this terminal open while using the app.

### 5. Open the frontend

Just open `frontend/index.html` in your browser. Double-click it or drag it into Chrome/Edge.

That's it.

---

## Endpoints

| Method | Endpoint | What it does |
|--------|----------|--------------|
| POST | /upload | Upload a .txt file, returns document ID and chunk count |
| POST | /ask | Send a question, get an answer with source chunk numbers |
| GET | /health | Check if the API is running |

---

## Error handling

| Situation | HTTP Code |
|-----------|-----------|
| Empty file | 400 |
| Wrong file type (not .txt) | 415 |
| Document ID not found | 404 |
| Empty question | 400 |
| LLM call fails | 503 |

---
