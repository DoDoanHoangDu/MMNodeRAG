import os
import pickle
import json
from LLM.qwen3_vl_reranker import Qwen3VLReranker
from tqdm import tqdm

#parameters
KNN = int(input("KNN: "))
REASONING = False

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
context_path = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}.jsonl")
question_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "sampled_questions.jsonl"))
oven_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "oven_images_sampled"))
eval_path = os.path.join(DIR_PATH, "data", f"evaluation_context_{KNN}nn{"_reasoning" if REASONING else ""}.jsonl")
print(question_path)
print(context_path)
print(eval_path)
input("Confirm?")

#load data
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

questions = {}
with open(question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        answer_eval = line["answer_eval"]
        if not isinstance(answer_eval, list):
            print(f"Invalid answer_eval for question {line['data_id']}: {answer_eval}")
            raise ValueError("Invalid answer_eval format")
        for i in range(len(answer_eval)):
            val = answer_eval[i]
            if not (isinstance(val, (str, int, float)) or (isinstance(val, dict) and "wikidata" in val and "range" in val)):
                print(f"Invalid answer_eval value for question {line['data_id']}: {val}")
                raise ValueError("Invalid answer_eval value format")
            if isinstance(val, dict):
                answer_eval[i]["value"] = answer_eval[i].pop("wikidata")
        questions[line["data_id"]] = (line["question"],line["image_id"], line["answer_eval"])

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
if os.path.exists(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
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

with open(eval_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        if qid in processed_questions:
            continue
        question = questions[qid][0]
        img_path = images[questions[qid][1]]
        answer_eval = questions[qid][2]

        text_query = f"Question: {question}\nAcceptable Answers: {answer_eval}"
        query = {"text": text_query, "image": img_path}
        context_nodes = contexts[qid]
        context_nodes_content = []
        for c in context_nodes:
            if "V" in c:
                c_content = {"image": nodes[c].content}
            else:
                c_content = {"text": nodes[c].content}
            context_nodes_content.append(c_content)

        inputs = {
            "instruction": "Given a question, its associated image and a list of acceptable answers. Score how useful the text or image is for answering the question.",
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