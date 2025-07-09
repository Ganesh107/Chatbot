import faiss

class VectorStore:
    def __init__(self, vectors, texts):
        self.texts = texts
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def search(self, query_vector, k=3):
        _, indices = self.index.search(query_vector, k)
        return [self.texts[i] for i in indices[0]]
