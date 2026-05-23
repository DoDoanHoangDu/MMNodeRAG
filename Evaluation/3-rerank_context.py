import os
import pickle
import json
from LLM.qwen3_vl_reranker import Qwen3VLReranker
from tqdm import tqdm
import pandas as pd
from PIL import Image
import ast
import io

#parameters
KNN = 8
REASONING = False

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
context_path = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}.jsonl")
questions0 = pd.read_parquet("Dataset/MMMU-Pro/test-00000-of-00002.parquet")
questions1 = pd.read_parquet("Dataset/MMMU-Pro/test-00001-of-00002.parquet")
questions = pd.concat([questions0, questions1], ignore_index=True)
reranked_path = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_reranked.jsonl")
print(context_path)
print(reranked_path)
input("Confirm?")

#load data
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

contexts = {}
with open(context_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        contexts[line["qid"]] = line["context_nodes"]

#load_progress:
processed_questions = set()
if os.path.exists(reranked_path):
    with open(reranked_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            processed_questions.add(line["qid"])

#reranker
model = Qwen3VLReranker(model_name_or_path="Qwen/Qwen3-VL-Reranker-2B")

#main loop:
def resize_image(image, max_size = 600):
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    scale = max_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.BILINEAR)

def sort_by_floats(strings, floats):
    paired = list(zip(strings, floats))
    paired.sort(key=lambda x: x[1], reverse=True)  # sort by float

    sorted_strings, sorted_floats = zip(*paired)
    return list(sorted_strings), list(sorted_floats)

with open(reranked_path, "a", encoding="utf-8") as f:
    for idx, row in tqdm(questions.iterrows()):
        qid = row["id"]
        if qid in processed_questions:
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


        query = {"text": question, "image": images}
        context_nodes = contexts[qid]
        context_nodes_content = []
        for c in context_nodes:
            if "V" in c:
                c_content = {"image": nodes[c].content}
            else:
                c_content = {"text": nodes[c].content[:20000]}
            context_nodes_content.append(c_content)
            
        inputs = {
            "instruction": "Retrieve images or text relevant to the user's query.",
            "query": query,
            "documents": context_nodes_content,
        }
        scores = model.process(inputs)

        context_nodes, scores = sort_by_floats(context_nodes, scores)
        data = {
            "qid": qid,
            "sorted_context_nodes": context_nodes,
            "sorted_relevance_scores": scores
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()