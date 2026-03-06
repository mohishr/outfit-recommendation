import base64
import numpy as np
from io import BytesIO

from flask import Flask, request, jsonify, render_template
from PIL import Image

from interface import OutfitCompatibilityAPI
from recommend import RecommenderEngine


app = Flask(__name__)


# ===============================
# Initialize ML Systems
# ===============================

print("Loading Compatibility Model...")
compatibility_api = OutfitCompatibilityAPI("./ggnn_ranker.weights.h5")

print("Loading Recommendation Engine...")
recommender = RecommenderEngine()


# ===============================
# Utility
# ===============================

def read_images(files):

    images = []

    for file in files:
        img = Image.open(file.stream).convert("RGB")
        arr = np.array(img)
        images.append(arr)

    return images


def extract_embeddings(images):

    embeddings = []

    for img in images:

        emb = compatibility_api.extractor.extract_from_array(img)

        embeddings.append(emb)

    return embeddings


# ===============================
# Home Page
# ===============================

@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# API 1: Outfit Compatibility
# ===============================

@app.route("/api/predict", methods=["POST"])
def predict_outfit():

    images = request.files.getlist("images")

    if len(images) < 2:
        return jsonify({"error": "Upload at least 2 images"}), 400

    try:

        img_arrays = read_images(images)

        score = compatibility_api.predict_from_arrays(img_arrays)

        # normalize score to 0-100
        score = 1 / (1 + np.exp(-score))
        score = float(score * 100)

        return jsonify({
            "compatibility_score": score
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ===============================
# API 2: Recommend Item
# ===============================

@app.route("/api/recommend_item", methods=["POST"])
def recommend_item():

    target_category = request.form.get("target_category")

    images = request.files.getlist("images")

    if not images:
        return jsonify({"error": "Upload outfit images"}), 400

    if not target_category:
        return jsonify({"error": "target_category missing"}), 400

    try:

        img_arrays = read_images(images)

        embeddings = extract_embeddings(img_arrays)

        partial_outfit = []

        for emb in embeddings:

            partial_outfit.append({
                "image_embedding": emb.tolist()
            })

        results = recommender.get_recommendations_for_outfit(
            partial_outfit=partial_outfit,
            target_category=target_category,
            top_n=5
        )

        return jsonify({
            "recommendations": results
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ===============================
# API 3: Virtual Stylist
# ===============================

@app.route("/api/generate_outfits", methods=["POST"])
def generate_outfits():

    try:

        images = request.files.getlist("images")

        if not images:
            return jsonify({"error": "Upload at least one item"}), 400

        img_arrays = read_images(images)

        embeddings = extract_embeddings(img_arrays)

        # Example simple stylist logic
        suggestions = []

        categories = ["Shoes", "Jacket", "Pants", "Hat"]

        for cat in categories:

            rec = recommender.get_recommendations_for_outfit(
                [{"image_embedding": e.tolist()} for e in embeddings],
                cat,
                top_n=1
            )

            if rec:
                suggestions.append(rec[0])

        return jsonify({
            "suggested_items": suggestions
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ===============================
# Health Check
# ===============================

@app.route("/api/health")
def health():
    return jsonify({"status": "running"})


# ===============================
# Run Server
# ===============================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )