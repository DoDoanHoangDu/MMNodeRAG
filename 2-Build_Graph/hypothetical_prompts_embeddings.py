from Node import Node
import torch
import faiss
import os
import pickle
import json
from LLM.qwen3_vl_embedding import Qwen3VLEmbedder
from LLM.prompts.hypothetical_prompts_generation_prompt import hypothetical_prompts_generation_prompt
from LLM.call_api import call_api
from tqdm import tqdm
import base64
import ast
import numpy as np

#hyper params
EMB_DIM = 2048
BATCH_SIZE = 8
print(f"Embedding dimension: {EMB_DIM}")

#file_paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
faiss_path = os.path.join(DIR_PATH, "data", "hype.faiss")
embedding_processed_ids_path = f"{DIR_PATH}/data/hype_embedding_processed_ids.txt"
hypothetical_prompts_path = os.path.join(DIR_PATH, "data", "hypothetical_prompts.jsonl")

#progress
hypothetical_prompts = {}
if os.path.exists(hypothetical_prompts_path):
    with open(hypothetical_prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            nid = line["nid"]
            prompts = line["hypothetical_prompts"]
            hypothetical_prompts[nid] = prompts

#validate llm list
def simple_strip(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:] # Remove ```json
    elif text.startswith("```"):
        text = text[3:] # Remove ```
    
    if text.endswith("```"):
        text = text[:-3] # Remove closing ```
    return text.strip()

def validate_list(l):
    if not isinstance(l, list):
        return False
    for i in l:
        if not isinstance(i, str):
            return False
        if not i.strip():
            return False
    return True

#data
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def encode_image(path):
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif"
    }.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime

with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

print("Total nodes to generate:", len([nid for nid in nodes if nodes[nid].node_type in {"S", "A", "H", "T", "V"}]))

MAX_ATTEMPTS = 20
with open(hypothetical_prompts_path, "a", encoding="utf-8") as f:
    for nid in tqdm(nodes):
        if nid in hypothetical_prompts:
            continue
        node = nodes[nid]
        if node.node_type not in {"S", "A", "H", "T", "V"}:
            continue
        elif node.node_type == "V":
            neighbor_entity_ids = [neighbor_id for neighbor_id in node.edges if nodes[neighbor_id].node_type == "N"]
            neighbor_entities = []
            for neighbor_id in neighbor_entity_ids:
                current_entities = nodes[neighbor_id].content
                if isinstance(current_entities, set):
                    current_entities = list(current_entities)
                elif isinstance(current_entities, str):
                    current_entities = [current_entities]
                if not isinstance(current_entities, list):
                    raise ValueError(f"Invalid entity content: {current_entities}")
                neighbor_entities.extend(current_entities)

            image_path = f"{BASE_PATH}/{node.content}"
            if not is_image_file(image_path):
                raise ValueError(f"Invalid image path: {image_path}")
            image, mime = encode_image(image_path)
            content = [
                {"type": "text", "text": f"[USER INPUT]\nThis image contains: {"; ".join(current_entities)}"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}},
            ]
        else:
            content = [{"type": "text", "text": f"[USER INPUT]\n{node.content}"}]
    
        for attempt in range(1,MAX_ATTEMPTS+1):
            response_text = None
            try:
                response_text, token = call_api(content=content, system_prompt=hypothetical_prompts_generation_prompt(), model="", mode="self-host")
                response = ast.literal_eval(simple_strip(response_text))
                if not validate_list(response) or len(response) == 0:
                    raise ValueError("Invalid response")
                hypothetical_prompts[nid] = response
                data = {"nid": nid, "hypothetical_prompts": response, "token": token}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                break
            except Exception as e:
                print(f"Attempt {attempt} failed for {nid}: {e}")
                print(response_text)
                if attempt == MAX_ATTEMPTS:
                    print(f"Failed on {nid}: {content}")
                    raise TimeoutError("Failed")

#embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
index = faiss.IndexFlatIP(EMB_DIM)

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

for nid in tqdm(hypothetical_prompts):
    prompts = hypothetical_prompts[nid]
    batch_embeddings = model.process(prompts).to(torch.float32).cpu().numpy()
    save_progress(np.vstack(batch_embeddings), [nid for _ in prompts])

with open(embedding_processed_ids_path, "r", encoding="utf-8") as f:
    embedding_processed_ids = list(line.strip() for line in f if line.strip())

if len(embedding_processed_ids) != index.ntotal:
    raise KeyError("Mismatched processed ids and index entries")