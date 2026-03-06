import json
import os
import pymongo
from pymongo import MongoClient

# Configurations (Update these paths if needed)
MONGO_URI = "mongodb://localhost:27021/"
DB_NAME = "fashion_recommendation_db"
COLLECTION_NAME = "Clothing_Items"

TRAIN_JSON_PATH = "test_no_dup_new_100.json"
CATEGORY_TXT_PATH = "category_id.txt"
IMAGE_VECTORS_DIR = "C:\\Users\\mohi_shr\\source\\my-repos\\Fashion-Recommendation-system-using-Graph-Neural-Networks-GNN-1\\data\\polyvore_image_vectors\\polyvore_image_vectors"
IMAGES_DIR_NAME = "polyvore-images" # Just used for constructing the image_url string

def load_categories(filepath):
    """
    Parses 'category_id.txt' into a dictionary mapping category_id (int) -> category_name (str)
    """
    cat_map = {}
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return cat_map
        
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                cat_id = int(parts[0])
                cat_name = parts[1]
                cat_map[cat_id] = cat_name
    return cat_map

def load_image_vector(set_id, item_index, vec_dir):
    """
    Reads the json array from <set_id>_<item_index>.json
    """
    filepath = os.path.join(vec_dir, f"{set_id}_{item_index}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                vector = json.load(f)
                return vector
        except json.JSONDecodeError:
            pass
    # Return None if not found, or zeros. We'll return None so we don't insert garbage.
    return None

def seed_database():
    print(f"Connecting to MongoDB at {MONGO_URI}...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Load Category Map
    print("Loading categories...")
    categories = load_categories(CATEGORY_TXT_PATH)
    
    # Load Training Data structure
    print(f"Loading outfit data from {TRAIN_JSON_PATH}...")
    if not os.path.exists(TRAIN_JSON_PATH):
        print(f"Error: Could not find {TRAIN_JSON_PATH}. Are you running from the 'data' directory?")
        return

    with open(TRAIN_JSON_PATH, 'r') as f:
        outfits = json.load(f)
        
    print(f"Found {len(outfits)} outfits. Preparing to seed database...")
    
    documents = []
    # To prevent inserting duplicates if items are shared across outfits, we'll keep track of what we processed.
    # Note: In polyvore data, items belong to specific sets, so <set_id>_<item_index> is unique.
    processed_items = set() 
    
    for outfit in outfits:
        set_id = outfit.get('set_id')
        items_cat = outfit.get('items_category', [])
        items_idx = outfit.get('items_index', [])
        
        for cat_id, idx in zip(items_cat, items_idx):
            item_key = f"{set_id}_{idx}"
            
            if item_key in processed_items:
                continue
            processed_items.add(item_key)
            
            # Fetch Image Vector
            # Some items might be missing their vectors in the directory, so we fetch safely
            img_vector = load_image_vector(set_id, idx, IMAGE_VECTORS_DIR)
            
            # Format Database Document
            cat_name = categories.get(cat_id, "Unknown Category")
            
            # Fetch Image Blob
            image_blob = None
            img_path = os.path.join(IMAGES_DIR_NAME, set_id, f"{idx}.jpg")
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as image_file:
                        image_blob = image_file.read()
                except Exception as e:
                    print(f"Failed to read image {img_path}: {e}")

            doc = {
                "_id": item_key,             # Use set_id_index as strict unique _id
                "user_id": None,             # Null for system catalog items
                "category_id": cat_id,
                "category": cat_name,
                "description": cat_name,     # Same as category as requested
                "image_blob": image_blob,    # Binary image format stored directly in DB
                "image_embedding": img_vector,
                "text_embedding": None       # Can be populated later when text vectors are extracted
            }
            
            documents.append(doc)
            
            # Batch insertion to save RAM/Network
            # Decreased batch size for large blob insertion to prevent 16MB document limits or mem issues
            if len(documents) >= 50:
                print(f"Inserting batch of {len(documents)} items...")
                # Use insert_many with ordered=False to skip duplicates if running script twice
                try:
                    collection.insert_many(documents, ordered=False)
                except pymongo.errors.BulkWriteError as bwe:
                    # Ignore Duplicate Key Errors (E11000)
                    pass
                documents = []

    # Insert remaining
    if len(documents) > 0:
        print(f"Inserting final batch of {len(documents)} items...")
        try:
            collection.insert_many(documents, ordered=False)
        except pymongo.errors.BulkWriteError as bwe:
            pass

    # Create indexes for faster queries on category
    print("Creating DB Indexes on 'category' and 'user_id'...")
    collection.create_index("category")
    collection.create_index("user_id")

    # Document Count Check
    count = collection.count_documents({})
    print(f"\nSeed complete! The '{COLLECTION_NAME}' collection now contains {count} items.")

if __name__ == "__main__":
    seed_database()
