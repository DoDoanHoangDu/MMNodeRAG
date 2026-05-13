import json
import pandas as pd
import os
import numpy as np
import torch
import faiss
from LLM.qwen3_vl_embedding import Qwen3VLEmbedder
import base64
from PIL import Image
import io
from tqdm import tqdm

EMB_DIM = 2048
BATCH_SIZE = 8
print(f"Embedding dimension: {EMB_DIM}")

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)
questions_faiss_path = os.path.join(DIR_PATH, "data", "questions.faiss")
questions_ids_path = os.path.join(DIR_PATH, "data", "questions_ids.txt")
processed_wikis_path = os.path.join(DIR_PATH, "data", "processed_wikis.jsonl")
wikidata_path = os.path.join(BASE_DIR, "Dataset", "Wiki6M_ver_1_0.jsonl")
questions_path = os.path.join(BASE_DIR, "Dataset", "MMMU-Pro")

#load questions
questions_df0 = pd.read_parquet(f"{questions_path}/test-00000-of-00002.parquet")
questions_df1 = pd.read_parquet(f"{questions_path}/test-00001-of-00002.parquet")
questions_df = pd.concat([questions_df0, questions_df1], ignore_index=True)

#embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

#load checkpoint
try:
    print("Loading Question Embeddings")
    with open(questions_ids_path, "r", encoding="utf-8") as f:
        embedding_processed_ids_list = [line.strip() for line in f if line.strip()]
    index = faiss.read_index(questions_faiss_path)
    id_count = len(embedding_processed_ids_list)
    vec_count = index.ntotal
    if id_count != vec_count:
        raise ValueError(f"Mismatch detected: IDs={id_count}, vectors={vec_count}")
    if len(questions_df) != vec_count:
        raise ValueError(f"Mismatch detected: questions={len(questions_df)}, vectors={vec_count}")
    if EMB_DIM != index.d:
        raise KeyError("Embeddings dimension mismatched")
    print("Questions Embeddings Loaded")
except Exception as e:
    print(f"Exception loading questions embeddings: {e}")
    print("Embedding Questions:")
    embedding_processed_ids_list = []
    index = faiss.IndexFlatIP(EMB_DIM)
    
    for idx, row in tqdm(questions_df.iterrows()):
        qid = row["id"]
        images = []
        for i in range(1,8):
            img_dict = row[f"image_{i}"]
            if not img_dict or not isinstance(img_dict, dict):
                continue
            img_data = img_dict["bytes"]
            
            try:
                image = Image.open(io.BytesIO(img_data)).convert("RGBA").convert("RGB")
            except Exception:
                if isinstance(img_data, bytes):
                    img_data = img_data.decode("utf-8")

                decoded = base64.b64decode(img_data)
                image = Image.open(io.BytesIO(decoded)).convert("RGBA").convert("RGB")
            images.append(image)
        content = [{"text": row["question"], "image": images, "instruction": "Represent the user's question"}]
        embeddings = model.process(content).to(torch.float32).cpu().numpy()
        embedding_processed_ids_list.append(qid)
        index.add(np.vstack(embeddings))
    
    faiss.write_index(index, questions_faiss_path)
    with open(questions_ids_path, "w", encoding="utf-8") as f:
        for qid in embedding_processed_ids_list:
            f.write(qid + "\n")

#load wiki
processed_wikis = set()
with open(processed_wikis_path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        record = json.loads(line)
        wiki_id = record["id"]
        processed_wikis.add(wiki_id)

with open(wikidata_path, "r", encoding="utf-8") as f, \
    open(processed_wikis_path, "a", encoding = "utf-8") as fout:
    for line in tqdm(f):
        record = json.loads(line)
        wiki_id = record["wikidata_id"]
        if wiki_id in processed_wikis:
            continue
        wiki_content = record["wikipedia_summary"]
        if not wiki_content or not isinstance(wiki_content, str):
            continue
        content = [{"instruction": "Represent the document for retrieval", "text": wiki_content}]
        embedding = model.process(content).to(torch.float32).cpu().numpy()
        similarity, _ = index.search(embedding, 1)

        data = {"id": wiki_id, "score": similarity[0][0]}
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        fout.flush()

#batch embeddings
batch_contents = []
batch_ids = []
with open(wikidata_path, "r", encoding="utf-8") as f, \
    open(processed_wikis_path, "a", encoding="utf-8") as fout:
    for line in tqdm(f):
        record = json.loads(line)
        wiki_id = record["wikidata_id"]
        if wiki_id in processed_wikis:
            continue
        wiki_content = record["wikipedia_summary"]
        if not wiki_content or not isinstance(wiki_content, str):
            continue
        batch_contents.append({"instruction": "Represent the document for retrieval", "text": wiki_content})
        batch_ids.append(wiki_id)

        # process batch
        if len(batch_contents) >= BATCH_SIZE:
            embeddings = model.process(batch_contents).to(torch.float32).cpu().numpy()
            similarities, _ = index.search(embeddings, 1)

            for wiki_id, similarity in zip(batch_ids, similarities):
                data = {"id": wiki_id, "score": float(similarity[0])}
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()
            # clear batch
            batch_contents = []
            batch_ids = []

    # process remaining items
    if batch_contents:
        embeddings = model.process(batch_contents).to(torch.float32).cpu().numpy()
        similarities, _ = index.search(embeddings, 1)
        for wiki_id, similarity in zip(batch_ids, similarities):
            data = {"id": wiki_id, "score": float(similarity[0])}
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        fout.flush()