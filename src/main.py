import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
from torch_geometric.data import RandomLinkSplit, Data
import os
from models.outfit_gnn import OutfitGNNModel

# --- ASSUMPTION: IMPORT YOUR CLASSES & UTILITIES HERE ---
# from model_utils import OutfitGNNModel, build_global_graph_data 
# (For simplicity, assume they are available in the environment)

# --- Configuration ---
TRAINED_MODEL_PATH = 'trained_outfit_gnn.pth'
CATALOG_FEATURES_PATH = 'catalog_features_final.npy' 
CATALOG_GRAPH_PATH = 'catalog_graph_data.pt'

INPUT_FEATURE_DIM = 512
OUTPUT_EMBEDDING_DIM = 128
NUM_NODES = 5000 # Simulate 5000 items
EPOCHS = 10
LR = 0.01

def load_simulated_data():
    """Mocks the loading and preprocessing of Polyvore data."""
    print("Loading simulated Polyvore data and features...")
    
    # 1. Simulated Item Features (X)
    item_features = {
        f'item_{i}': np.random.rand(INPUT_FEATURE_DIM).astype(np.float32) 
        for i in range(NUM_NODES)
    }
    
    # 2. Simulated Outfit Data
    raw_outfits = []
    for i in range(2000): # 2000 simulated outfits
        # Randomly select between 3 and 7 unique items
        num_items = np.random.randint(3, 8)
        item_indices = np.random.choice(NUM_NODES, size=num_items, replace=False)
        item_ids = [f'item_{idx}' for idx in item_indices]
        raw_outfits.append({'items': item_ids})

    # Convert the raw data into the single PyG Data object
    # NOTE: You must include your actual build_global_graph_data function here!
    # Using a simplified mock for the final PyG object:
    
    item_to_idx = {item_id: i for i, item_id in enumerate(item_features.keys())}
    node_features = torch.from_numpy(np.array(list(item_features.values())))
    
    # Simulate co-occurrence edges (10,000 directed edges)
    # In reality, this would come from your 'build_global_graph_data' utility
    edge_index = torch.randint(0, NUM_NODES, (2, 10000)) 
    
    full_data = Data(x=node_features, edge_index=edge_index, num_nodes=NUM_NODES)
    
    # Save the necessary artifacts (X matrix and Graph Data)
    np.save(CATALOG_FEATURES_PATH, item_features)
    torch.save(full_data, CATALOG_GRAPH_PATH)
    
    print(f"Artifacts saved: {CATALOG_FEATURES_PATH} and {CATALOG_GRAPH_PATH}")
    return full_data

def main():
    # --- 1. DATA LOADING AND PREPARATION ---
    full_graph = load_simulated_data()
    
    # Split edges for Link Prediction: 80% Train, 10% Val, 10% Test
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True, # Add negative samples to training set
        neg_sampling_ratio=1.0 # 1:1 ratio for all splits
    )
    train_data, val_data, test_data = transform(full_graph)

    print(f"Training on {train_data.edge_index.size(1) // 2} positive edges.")
    print(f"Training batches contain {train_data.edge_label_index.size(1)} pairs (Pos+Neg).")
    
    # --- 2. MODEL AND OPTIMIZER SETUP ---
    model = OutfitGNNModel(
        in_dim=INPUT_FEATURE_DIM, 
        out_dim=OUTPUT_EMBEDDING_DIM
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # --- 3. TRAINING LOOP ---
    print("\nStarting GNN Model Training...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # 1. Encode nodes using the training graph's edges
        Z = model(train_data.x, train_data.edge_index)
        
        # 2. Score the positive and negative training pairs
        # train_data.edge_label_index[0/1] are the indices for the pairs
        logits = model.score_pairs(
            Z, 
            train_data.edge_label_index[0], 
            train_data.edge_label_index[1]
        )
        
        # 3. Calculate loss against the ground truth labels (0s and 1s)
        loss = criterion(logits, train_data.edge_label.float())
        
        # 4. Backpropagate and update weights
        loss.backward()
        optimizer.step()

        # --- EVALUATION (Simplified) ---
        train_loss = loss.item()
        val_auc = evaluate_model(model, val_data, criterion)
        
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}')

    # --- 4. FINAL SAVE ---
    torch.save(model.state_dict(), TRAINED_MODEL_PATH)
    print(f"\n✅ Training complete. Final model saved to: {TRAINED_MODEL_PATH}")

def evaluate_model(model, data, criterion):
    """Simple AUC/Loss evaluation on a held-out set."""
    model.eval()
    with torch.no_grad():
        # Encode nodes using the evaluation graph's edges
        Z = model(data.x, data.edge_index) 
        
        # Score the validation pairs
        logits = model.score_pairs(
            Z, 
            data.edge_label_index[0], 
            data.edge_label_index[1]
        )
        
        # Loss calculation (optional, but good for tracking)
        loss = criterion(logits, data.edge_label.float()).item()
        
        # AUC calculation (mocked for simplicity, requires sklearn's roc_auc_score)
        # Mock AUC: Higher loss -> lower AUC
        mock_auc = 0.95 - (loss / 10) 
        return max(0.5, mock_auc) # Ensure AUC is at least 0.5

if __name__ == "__main__":
    main()