# src/similarity.py
import faiss, numpy as np

class CosineIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []

    def add(self, vectors, ids):
        self.index.add(vectors)  # vectors should be L2-normalized
        self.ids.extend(ids)

    def search(self, query_vector, k=10):
        D, I = self.index.search(query_vector[None, :], k)
        return [(self.ids[i], float(D[0][j])) for j, i in enumerate(I[0])]