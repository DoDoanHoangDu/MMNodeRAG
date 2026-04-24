import os
import json
import time
from tqdm import tqdm
import faiss
import pickle
import torch
from LLM.qwen3_vl_embedding import Qwen3VLEmbedder

#hyper parameters
QUESTION_LEVEL = int(input("Question Level: "))
KNN = int(input("KNN: "))

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
question_path = os.path.normpath(os.path.join(DIR_PATH, "data", "subquestions.jsonl"))
oven_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "oven_images_sampled"))
output_path = f"{DIR_PATH}/data/{KNN}nn_{QUESTION_LEVEL}.jsonl"
print(question_path)
print(output_path)
input("Confirm?")

#load questions
start = time.time()
#load subquestions at a level, merge with answers of previous levels
questions = {}
with open(question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        subquestions = line["subquestions"]
        if len(subquestions) < QUESTION_LEVEL+2:
            continue
        elif len(subquestions) == QUESTION_LEVEL+2:
            questions[line["qid"]] = (line["question"], line["image_id"])
        else:
            questions[line["qid"]] = (subquestions[QUESTION_LEVEL],line["image_id"])
    if len(questions) == 0:
        raise RuntimeError("No questions left to decompose")
    print(f"Number of questions: {len(questions)}")

#load previous answers
for level in range(QUESTION_LEVEL-1,-1, -1):
    previous_answer_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"answers_{KNN}nn_{level}.jsonl"))
    if os.path.exists(previous_answer_path):
        prev_answer_count = 0
        with open(previous_answer_path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                qid = line["qid"]
                answer = line["answer"]
                questions[qid] = (answer + "\n" + questions[qid][0], questions[qid][1])
                prev_answer_count += 1
        if len(questions) != prev_answer_count:
            raise RuntimeError(f"Previous answers and questions at level {level} mismatch: {prev_answer_count}/{len(questions)}")
    else:
        raise RuntimeError(f"Previous answers not exist for question level: {level}")

#load images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def extract_id(filename):
    return os.path.splitext(filename)[0]

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

images = {}
for file in os.listdir(oven_path):
    if is_image_file(file):
        images[extract_id(file)] = os.path.join(oven_path, file)
    else:
        print(f"Invalid file: {file}")

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
with open(output_path,"a", encoding = "utf-8") as f:
    for qid in tqdm(questions):
        if qid in processed_qids:
            continue
        question = questions[qid][0]
        image = images[questions[qid][1]]
        full_question = [{"text": question, "image": image, "instruction": "Retrieve images or text relevant to the user's query."}]
        query_embedding = model.process(full_question).to(torch.float32).cpu().numpy()
        similarity, idx = hnsw.search(query_embedding, KNN)
        embedding_node_ids = [embedding_ids[i] for i in idx[0]]
        data = {"qid": qid, "knn": embedding_node_ids}
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()