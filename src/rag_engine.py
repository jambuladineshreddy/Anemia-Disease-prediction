import os
import sys
import glob
import re
import numpy as np

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MEDICAL_DOCS_DIR


class MedicalRAGEngine:
    def __init__(self, docs_dir=None):
        self.docs_dir = docs_dir or MEDICAL_DOCS_DIR
        self.documents = []
        self.chunks = []
        self.use_chroma = False
        self._load_and_chunk_documents()
        self._initialize_retriever()

    def _load_and_chunk_documents(self):
        """Loads medical markdown files and breaks them into logical section chunks."""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            
        doc_files = glob.glob(os.path.join(self.docs_dir, "*.md")) + glob.glob(os.path.join(self.docs_dir, "*.txt"))
        
        for file_path in doc_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Split document by headers (e.g., # or ##)
            raw_sections = re.split(r'\n(?=##?\s+)', content)
            for idx, sec in enumerate(raw_sections):
                cleaned = sec.strip()
                if len(cleaned) > 50:
                    self.chunks.append({
                        "id": f"{os.path.basename(file_path)}_chunk_{idx}",
                        "source": os.path.basename(file_path),
                        "text": cleaned
                    })

    def _initialize_retriever(self):
        """Initializes fast, offline TF-IDF semantic matching with optional Chroma vector store."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        if self.chunks:
            texts = [c["text"] for c in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Optional neural embedding integration if specifically enabled via env var
        if os.getenv("ENABLE_NEURAL_RAG", "0") == "1":
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer
                self.chroma_client = chromadb.Client()
                self.collection = self.chroma_client.get_or_create_collection(name="medical_guidelines")
                self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                if self.chunks and self.collection.count() == 0:
                    texts = [c["text"] for c in self.chunks]
                    metadatas = [{"source": c["source"]} for c in self.chunks]
                    ids = [c["id"] for c in self.chunks]
                    embeddings = self.embedder.encode(texts).tolist()
                    self.collection.add(
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                self.use_chroma = True
            except Exception:
                self.use_chroma = False

    def retrieve(self, query, top_k=2):
        """Retrieves top_k relevant medical guidelines for a given query."""
        if not self.chunks:
            return [{"source": "system", "text": "No medical guidelines uploaded."}]
            
        if self.use_chroma:
            try:
                query_embedding = self.embedder.encode([query]).tolist()
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=top_k
                )
                retrieved = []
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    retrieved.append({
                        "source": meta["source"],
                        "text": doc
                    })
                return retrieved
            except Exception:
                pass
                
        # Fast, offline cosine similarity retrieval using TF-IDF
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved = []
        for idx in top_indices:
            retrieved.append({
                "source": self.chunks[idx]["source"],
                "text": self.chunks[idx]["text"],
                "score": float(scores[idx])
            })
        return retrieved

    def retrieve_for_patient(self, shap_result, top_k=2):
        """Synthesizes patient clinical search query based on prediction & SHAP features."""
        morphology = shap_result.get("morphology", "")
        anomalies = ", ".join(shap_result.get("anomalies", []))
        top_features = [f["feature"] for f in shap_result.get("feature_attributions", [])[:2]]
        
        query = f"Anemia diagnosis guidelines for {morphology} anemia. Patient anomalies: {anomalies}. Key drivers: {', '.join(top_features)}."
        return self.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    rag = MedicalRAGEngine()
    res = rag.retrieve("Microcytic hypochromic anemia low hemoglobin low MCV iron deficiency", top_k=2)
    print("--- RAG Retrieval Output ---")
    for doc in res:
        print(f"Source: {doc['source']}\nText Excerpt: {doc['text'][:150]}...\n")
