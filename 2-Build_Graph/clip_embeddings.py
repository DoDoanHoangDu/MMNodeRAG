from Node import Node
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import faiss
import os
import pickle
import json

embed_content = {}

#load S,A,T nodes
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)

g2_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g1.pkl")
with open(g2_path, "rb") as f:
    nodes = pickle.load(f)
for node in nodes.values():
    if node.node_type not in ["S", "A", "T"]:
        continue
    embed_content[node.node_id] = node.content

#load H content
communities_path = os.path.join(DIR_PATH, "data", "communities.jsonl")
with open(communities_path, "w", encoding="utf-8") as f:
    for line in f:
        line = json.loads(f)
        embed_content[line["community_id"]] = line["summary"]

#embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = 

def get_embedding(data_dict):
    ids = list(data_dict.keys())
    texts = list(data_dict.values())
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    return ids, embeddings

ids, embeddings = get_embedding(embed_content)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"Embedding dimension: {dimension}")

#save progress
faiss.write_index(index, os.path.join(DIR_PATH, "data", "embeddings.faiss"))
with open(os.path.join(DIR_PATH, "data", "embedding_ids.json"), "w") as f:
    json.dump(ids, f)