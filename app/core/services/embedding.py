import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings

#embedder = SentenceTransformer(settings.EMBED_MODEL)
# offline model:


local_model_path = (
    "/root/.cache/huggingface/hub/models--heydariAI--persian-embeddings/snapshots/9fca7a60453166798bf59dc9869da2202da1b0c0"
)
embedder = SentenceTransformer(local_model_path)



def get_embedding(text: str):
    return embedder.encode(text, normalize_embeddings=True).tolist()

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)
