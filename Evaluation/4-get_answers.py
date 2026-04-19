import os
import pickle
import json
from LLM.prompts.answer_prompt import answer_prompt
from tqdm import tqdm

#parameters
KNN = 16
REASONING = False

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
question_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "sampled_questions.jsonl"))
oven_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "oven_images_sampled"))
reranked_path = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_reranked.jsonl")
print(question_path)
print(reranked_path)
input("Confirm?")

#load data
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

questions = {}
with open(question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        questions[line["data_id"]] = (line["question"],line["image_id"])

contexts = {}
with open(reranked_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        current_context = []
        for i in range(len(line["sorted_context_nodes"])):
            current_context.append((line["sorted_context_nodes"][i], line["sorted_relevance_scores"][i]))
        contexts[line["qid"]] = line["context_nodes"]

#load images
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}

images = {}
for img in os.listdir(oven_path):
    img_id, ext = img.split(".")
    if ext.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Not image: {img}")
    images[img_id] = f"{oven_path}/{img}"

#load_progress:
processed_questions = set()
if os.path.exists(reranked_path):
    with open(reranked_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            processed_questions.add(line["qid"])