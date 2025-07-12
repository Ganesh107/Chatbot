from app.data_loader import load_products
from app.embedder import Embedder
from app.vector_store import VectorStore
import numpy as np
import requests
from app.constants import Constants 

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
            Constants.llm_endpoint,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        response = response.json()["response"]
        return response if "context" not in response else Constants.invalid_context
    
    def answer_question(self, question):
        q_vec = np.array([self.embedder.encode([question])[0]])
        top_contexts = self.vstore.search(q_vec)
        context = "\n".join(top_contexts)
        if not context:
            return Constants.invalid_context
        prompt = self.build_prompt(context, question)
        return self.call_mistral(prompt)
