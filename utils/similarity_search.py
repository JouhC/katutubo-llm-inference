import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class UPVectorDB:
    def __init__(self):
        self.encoder_id = os.getenv("ENCODER_ID")
        self.qdrant_path = os.getenv("QDRANT_PATH")
        self._validate_env_vars()

        self.db = QdrantClient(path=self.qdrant_path)
        self.encoder = SentenceTransformer(self.encoder_id)

    def _validate_env_vars(self):
        if not self.encoder_id:
            raise EnvironmentError("ENCODER_ID environment variable is not set.")
        if not self.qdrant_path:
            raise EnvironmentError("QDRANT_PATH environment variable is not set.")

    def similarity_search(self, prompt: str):
        try:
            vector = self.encoder.encode(prompt).tolist()
            hits = self.db.search(
                collection_name="up_FAQs",
                query_vector=vector,
                score_threshold=0.6,
                limit=1
            )

            if not hits:
                return None

            return hits[0].payload.get('Answer')  # Ensures we access the first result safely
        except Exception as e:
            raise RuntimeError(f"Similarity search failed: {str(e)}")
