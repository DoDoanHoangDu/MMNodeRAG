import os
import pickle
import faiss
from tqdm import tqdm
import Node

# Paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)

g4_path = os.path.join(BASE_PATH, "2-Build_Graph", "data", "g4.pkl")
embedding_index_path = os.path.join(BASE_PATH, "2-Build_Graph", "data", "embeddings_hnsw.faiss")
embedding_ids_path = os.path.join(BASE_PATH, "2-Build_Graph", "data", "embedding_processed_ids.txt")

output_dir = os.path.join(DIR_PATH, "data")
os.makedirs(output_dir, exist_ok=True)

# Load original FAISS index
hnsw = faiss.read_index(embedding_index_path)

# Load embedding IDs
with open(embedding_ids_path, "r", encoding="utf-8") as f:
    embedding_ids = [line.strip() for line in f]

# Load graph nodes
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

# Build T-only HNSW index
M = 32
dimension = hnsw.d  # Should be 2048
t_index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
t_index.hnsw.efConstruction = 200
t_index.hnsw.efSearch = 64

t_ids = []

print("Building T-only index...")

for i, nid in tqdm(enumerate(embedding_ids), total=len(embedding_ids)):
    if nodes[nid].node_type == "T":
        vec = hnsw.reconstruct(i).reshape(1, -1)
        t_index.add(vec)
        t_ids.append(nid)

# Save T-only FAISS index
faiss.write_index(
    t_index,
    os.path.join(output_dir, "t_embeddings_hnsw.faiss")
)

# Save corresponding IDs
with open(os.path.join(output_dir, "t_ids.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(t_ids))

print(f"Done!")
print(f"T nodes: {len(t_ids)}")
print(f"FAISS index saved to: {os.path.join(output_dir, 't_embeddings_hnsw.faiss')}")
print(f"ID mapping saved to: {os.path.join(output_dir, 't_ids.txt')}")