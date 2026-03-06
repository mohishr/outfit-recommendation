# recommend.py

import numpy as np
import pymongo
import base64
from interface import OutfitCompatibilityAPI

MONGO_URI = "mongodb://host.docker.internal:27021/"
DB_NAME = "fashion_recommendation_db"
COLLECTION_NAME = "Clothing_Items"


class RecommenderEngine:

    def __init__(self):

        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]

        self.predictor = OutfitCompatibilityAPI()

        print("Connected to MongoDB")

    def get_recommendations_for_outfit(self, partial_outfit, target_category, top_n=5):

        if not partial_outfit:
            return []

        candidates = list(self.collection.find(
            {"category": target_category},
            {"_id": 1, "category": 1, "description": 1, "image_embedding": 1, "image_blob": 1}
        ))

        if not candidates:
            return []

        scored = []

        outfit_embeddings = [
            np.array(x["image_embedding"], dtype=np.float32)
            for x in partial_outfit
        ]

        print(f'candidates {len(candidates)}')

        for candidate in candidates:

            if not candidate.get("image_embedding"):
                continue

            candidate_embedding = np.array(
                candidate["image_embedding"], dtype=np.float32
            )

            try:

                embeddings = outfit_embeddings + [candidate_embedding]

                raw_score = self.predictor.predict_from_embeddings(embeddings)

                normalized_score = max(0, min(100, raw_score * 10))

                scored.append({
                    "item_id": str(candidate["_id"]),
                    "category": candidate["category"],
                    "description": candidate.get("description", ""),
                    "score": float(normalized_score),
                    "image_base64": base64.b64encode(
                        candidate["image_blob"]
                    ).decode("utf-8") if candidate.get("image_blob") else None
                })

            except Exception as e:
                print("Prediction error:", e)

        scored.sort(key=lambda x: x["score"], reverse=True)

        return scored[:top_n]