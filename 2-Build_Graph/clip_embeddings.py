from Node import Node
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
import os
import pickle
import json
from PIL import Image

text_data = {}
image_data = {}

#load S,A,T nodes
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)

g2_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g2.pkl")
with open(g2_path, "rb") as f:
    nodes = pickle.load(f)
for node in nodes.values():
    if node.node_type in ["S", "A", "T"]:
        text_data[node.node_id] = node.content
    elif node.node_type == "V":
        image_data[node.node_id]=Image.open(f"{BASE_PATH}/{node.content}").convert("RGB"),

#load H content
communities_path = os.path.join(DIR_PATH, "data", "communities.jsonl")
with open(communities_path, "w", encoding="utf-8") as f:
    for line in f:
        line = json.loads(f)
        text_data[line["community_id"]] = line["summary"]

#embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_embedding(data_dict):
    ids = list(data_dict.keys())
    texts = list(data_dict.values())
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    return ids, embeddings

ids, embeddings = get_embedding(text_data)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"Embedding dimension: {dimension}")

#save progress
faiss.write_index(index, os.path.join(DIR_PATH, "data", "embeddings.faiss"))
with open(os.path.join(DIR_PATH, "data", "embedding_ids.json"), "w") as f:
    json.dump(ids, f)