from app.data_loader import load_products
from app.embedder import Embedder
from app.vector_store import VectorStore
import numpy as np
import requests

class RAGEngine:
    def __init__(self):
        self.embedder = Embedder()
        self.contexts = load_products()
        self.vectors = np.array(self.embedder.encode(self.contexts))
        self.vstore = VectorStore(self.vectors, self.contexts)

    def build_prompt(self, context, question):
        return f"""You are a helpful assistant. Use the context below to answer the question.
        Context:
        {context}
        Question:
        {question}
        Answer:"""

    def call_mistral(self, prompt):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        response = response.json()["response"]
        return response if "The context provided does not" not in response else "I'm sorry, I don't have enough information to answer that question."
    
    def answer_question(self, question):
        q_vec = np.array([self.embedder.encode([question])[0]])
        top_contexts = self.vstore.search(q_vec)
        context = "\n".join(top_contexts)
        prompt = self.build_prompt(context, question)
        return self.call_mistral(prompt)
