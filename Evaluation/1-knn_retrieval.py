import os
import json
import time
from tqdm import tqdm
import faiss
import pickle
import torch
from LLM.qwen3_vl_embedding import Qwen3VLEmbedder
import pandas as pd
from PIL import Image
import io
import ast

K = 16

#paths
start = time.time()
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
questions0 = pd.read_parquet("Dataset/MMMU-Pro/test-00000-of-00002.parquet")
questions1 = pd.read_parquet("Dataset/MMMU-Pro/test-00001-of-00002.parquet")
questions = pd.concat([questions0, questions1], ignore_index=True)
output_path = f"{DIR_PATH}/data/knn.jsonl"
print(output_path)
input("Confirm?")

#load embeddings
hnsw = faiss.read_index(f"{BASE_PATH}/2-Build_Graph/data/embeddings_hnsw.faiss")
with open(f"{BASE_PATH}/2-Build_Graph/data/embedding_processed_ids.txt", "r") as f:
    embedding_ids = [line.strip() for line in f]
embeddings = hnsw.reconstruct_n(0, hnsw.ntotal)

#load graph
with open(f"{BASE_PATH}/2-Build_Graph/data/g4.pkl", "rb") as f:
    graph = pickle.load(f)
print(f"Data load time: {time.time() - start:.2f} seconds.")

# load embedding model
start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
print("Model loaded")
print(f"Model load time: {time.time() - start:.2f} seconds.")

#load checkpoint
processed_qids = set()
if os.path.exists(output_path):
    with open(output_path,"r", encoding = "utf-8") as f:
        for line in f:
            line = json.loads(line)
            processed_qids.add(line["qid"])

#run loop
def resize_image(image, max_size = 1000):
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    scale = max_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.LANCZOS)

with open(output_path,"a", encoding = "utf-8") as f:
    for idx, row in tqdm(questions.iterrows()):
        qid = row["id"]
        if qid in processed_qids:
            continue
        question = row["question"]
        options = ast.literal_eval(row["options"])
        for i, item in enumerate(options):
            label = chr(ord('A') + i)
            question += "\n" + f"{label}) {item}"
        images = []
        for i in range(1,8):
            img_dict = row[f"image_{i}"]
            if not img_dict or not isinstance(img_dict, dict):
                continue
            img_data = img_dict["bytes"]
            image = Image.open(io.BytesIO(img_data)).convert("RGBA").convert("RGB")
            image = resize_image(image)
            images.append(image)

        full_question = [{"text": question, "image": images, "instruction": "Retrieve images or text relevant to the user's query."}]
        query_embedding = model.process(full_question).to(torch.float32).cpu().numpy()
        similarity, idx = hnsw.search(query_embedding, K)
        embedding_node_ids = [embedding_ids[i] for i in idx[0]]
        data = {"qid": qid, "knn": embedding_node_ids}
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()