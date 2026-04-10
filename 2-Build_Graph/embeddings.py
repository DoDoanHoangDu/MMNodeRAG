from Node import Node
import numpy as np
import torch
import faiss
import os
import pickle
import json
from LLM.qwen3_vl_embedding import Qwen3VLEmbedder
from tqdm import tqdm
import time
#hyper params
EMB_DIM = 2048
BATCH_SIZE = 8
print(f"Embedding dimension: {EMB_DIM}")

#file_paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g2_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g2.pkl")
faiss_path = os.path.join(DIR_PATH, "data", "embeddings.faiss")
embedding_processed_ids_path = f"{DIR_PATH}/data/embedding_processed_ids.txt"

#load checkpoint
if os.path.exists(embedding_processed_ids_path):
    with open(embedding_processed_ids_path, "r", encoding="utf-8") as f:
        embedding_processed_ids_list = [line.strip() for line in f if line.strip()]
    embedding_processed_ids = set(embedding_processed_ids_list)
else:
    embedding_processed_ids_list = []
    embedding_processed_ids = set()

if os.path.exists(faiss_path):
    index = faiss.read_index(faiss_path)
    #embeddings_list = list(index.reconstruct_n(0, index.ntotal))
else:
    index = faiss.IndexFlatIP(EMB_DIM)
    #embeddings_list = []

#recovery
id_count = len(embedding_processed_ids_list)
vec_count = index.ntotal
if id_count != vec_count:
    print(f"Mismatch detected: IDs={id_count}, vectors={vec_count}")
    if vec_count > id_count:
        print("Truncating FAISS index...")
        if id_count == 0:
            index = faiss.IndexFlatIP(EMB_DIM)
        else:
            vectors = index.reconstruct_n(0, id_count)
            new_index = faiss.IndexFlatIP(EMB_DIM)
            new_index.add(vectors)
            index = new_index
        faiss.write_index(index, faiss_path)
    else:
        print("Truncating ID file...")
        embedding_processed_ids_list = embedding_processed_ids_list[:vec_count]

        with open(embedding_processed_ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(embedding_processed_ids_list) + "\n")

        embedding_processed_ids = set(embedding_processed_ids_list)

if EMB_DIM != index.d:
    raise KeyError("Embeddings dimension mismatched")
print(f"Checkpoint loaded: {index.ntotal} vectors embedded at dimension {index.d}")

#load new data to embed
data = []
embedding_ids = []

#load S,A,T,V nodes
def is_image_file(path):
    if not os.path.isfile(path):
        return False
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
    return path.lower().endswith(valid_extensions)

with open(g2_path, "rb") as f:
    nodes = pickle.load(f)
for node in nodes.values():
    if node.node_id in embedding_processed_ids:
        continue
    if node.node_type in ["S", "A", "T"]:
        content = {"text": node.content, "instruction": "Represent the document for retrieval."}
    elif node.node_type == "V":
        image_path = f"{BASE_PATH}/{node.content}"
        if not is_image_file(image_path):
            raise ValueError(f"Invalid image path: {image_path}")
        content = {"image": image_path, "instruction": "Represent the image for retrieval."}
    else:
        continue
    data.append(content)
    embedding_ids.append(node.node_id)


#load H content
communities_path = os.path.join(DIR_PATH, "data", "communities.jsonl")
with open(communities_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        if line["community_id"] in embedding_processed_ids:
            continue
        content = {"text": line["summary"], "instruction": "Represent the document for retrieval."}
        data.append(content)
        embedding_ids.append(line["community_id"])

if len(embedding_ids) != len(set(embedding_ids)):
    raise KeyError("Duplicate data loaded")

#embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

#run loop
def save_progress(vectors, ids):
    if vectors.shape[0] != len(ids):
        raise KeyError("Mismatched progress to save")
    with open(embedding_processed_ids_path, "a", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
        f.flush()
        os.fsync(f.fileno())
    index.add(vectors)
    faiss.write_index(index, faiss_path)


pending_vectors = []
pending_ids = []
start = time.time()
for i in tqdm(range(0,len(data), BATCH_SIZE)):
    batch = data[i:i+BATCH_SIZE]
    batch_ids = embedding_ids[i:i+BATCH_SIZE]
    batch_embeddings = model.process(batch).to(torch.float32).cpu().numpy()

    pending_ids.extend(batch_ids)
    pending_vectors.append(batch_embeddings)

    if time.time() - start >= 300:
        save_progress(np.vstack(pending_vectors),pending_ids)
        pending_vectors, pending_ids = [], []
        start = time.time()

if pending_ids:
    save_progress(np.vstack(pending_vectors), pending_ids)

print("Completed")