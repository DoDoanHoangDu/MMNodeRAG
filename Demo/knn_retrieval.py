import os
import torch

#load images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def extract_id(filename):
    return os.path.splitext(filename)[0]

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def knn_retrieval(model, hnsw, embedding_ids, K, question, image_path):
    full_question = [{"text": question, "image": image_path, "instruction": "Retrieve images or text relevant to the user's query."}]
    query_embedding = model.process(full_question).to(torch.float32).cpu().numpy()
    similarity, idx = hnsw.search(query_embedding, K)
    embedding_node_ids = [embedding_ids[i] for i in idx[0]]
    return embedding_node_ids