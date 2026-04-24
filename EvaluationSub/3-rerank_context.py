import os
import pickle
import json
from LLM.qwen3_vl_reranker import Qwen3VLReranker
from tqdm import tqdm

#parameters
QUESTION_LEVEL = int(input("Question Level: "))
KNN = int(input("KNN: "))
REASONING = False

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
context_path = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_{QUESTION_LEVEL}.jsonl")
question_path = os.path.normpath(os.path.join(DIR_PATH, "data", "subquestions.jsonl"))
oven_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "oven_images_sampled"))
reranked_path = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_reranked_{QUESTION_LEVEL}.jsonl")
print(question_path)
print(context_path)
print(reranked_path)
input("Confirm?")

#load data
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

#load subquestions at a level, merge with answers of previous levels
questions = {}
with open(question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        subquestions = line["subquestions"]
        if len(subquestions) < QUESTION_LEVEL:
            continue
        elif len(subquestions) == QUESTION_LEVEL:
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

contexts = {}
with open(context_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
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

#reranker
model = Qwen3VLReranker(model_name_or_path="Qwen/Qwen3-VL-Reranker-2B")

#main loop:
def sort_by_floats(strings, floats):
    paired = list(zip(strings, floats))
    paired.sort(key=lambda x: x[1], reverse=True)  # sort by float

    sorted_strings, sorted_floats = zip(*paired)
    return list(sorted_strings), list(sorted_floats)

with open(reranked_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        if qid in processed_questions:
            continue
        question = questions[qid][0]
        img_path = images[questions[qid][1]]
        query = {"text": question, "image": img_path}
        context_nodes = contexts[qid]
        context_nodes_content = []
        for c in context_nodes:
            if "V" in c:
                c_content = {"image": nodes[c].content}
            else:
                c_content = {"text": nodes[c].content}
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