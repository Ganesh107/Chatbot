from fastapi import FastAPI, Request
from app.rag_engine import RAGEngine

app = FastAPI()
rag = RAGEngine()

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    answer = rag.answer_question(question)
    return {"question": question, "answer": answer}
