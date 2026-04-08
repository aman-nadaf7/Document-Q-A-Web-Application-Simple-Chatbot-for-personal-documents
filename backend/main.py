import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from models import UploadResponse, AskRequest, AskResponse, HealthResponse
from rag import embed_and_store, retrieve_relevant_chunks, document_exists, EMBEDDING_MODEL_NAME

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
app = FastAPI(title="Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_llm():
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured in environment.")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0,
    )



@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Accept a .txt file, chunk it, embed it, and store it."""

    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=415,
            detail="Only .txt files are supported. Please upload a plain text file."
        )

    content_bytes = await file.read()

    if not content_bytes or not content_bytes.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty. Please upload a file with content."
        )

    text = content_bytes.decode("utf-8", errors="replace")

    document_id, total_chunks = embed_and_store(text, file.filename)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        total_chunks=total_chunks,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Answer a question using only the content of the uploaded document."""

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if not document_exists(request.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{request.document_id}' not found. Please upload a document first."
        )

    relevant_chunks, source_numbers = retrieve_relevant_chunks(
        request.document_id, request.question, top_k=3
    )

    context = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = """You are a document question-answering assistant.
Your ONLY job is to answer questions based strictly on the provided document context.

CRITICAL RULES:
1. Answer ONLY from the provided context below.
2. If the answer is NOT in the context, say exactly: "The answer to this question is not found in the uploaded document."
3. Do NOT use your general knowledge. Do NOT guess. Do NOT infer beyond what is written.
4. Be concise and direct.
5. Quote or reference specific parts of the context when possible."""

    user_message = f"""Document Context:
{context}

Question: {request.question}

Answer based only on the above context:"""

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        answer = response.content.strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM service error: {str(e)}"
        )

    return AskResponse(answer=answer, sources=source_numbers)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return API status and model information."""
    return HealthResponse(
        status="ok",
        llm_model=LLM_MODEL_NAME,
        embedding_model=EMBEDDING_MODEL_NAME,
    )
