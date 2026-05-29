import os
import pickle
import json
from LLM.prompts.answer_prompt import answer_prompt
from LLM.prompts.cot_answer_prompt import cot_answer_prompt
from LLM.call_api import call_api
from tqdm import tqdm
import base64
import time

#parameters
KNN = int(input("KNN: "))
COT = True if input("Chain-of-thought instruction (y/n): ").lower() == "y" else False
REASONING = False
LIMIT = 0.5

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
question_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "sampled_questions.jsonl"))
subquestions_path = os.path.normpath(os.path.join(DIR_PATH, "data", "subquestions.jsonl"))
oven_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "oven_images_sampled"))
reranked_path_sub = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_reranked.jsonl")
reranked_path = os.path.join(BASE_PATH, "Evaluation", "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_reranked.jsonl")
output_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"answers_{KNN}nn{'_COT' if COT else ''}.jsonl"))
print(question_path)
print(reranked_path_sub)
print(output_path)
input("Confirm?")

#load data
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

questions = {}
answer_eval = {}
with open(question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        questions[line["data_id"]] = (line["question"],line["image_id"])
        answer_eval[line["data_id"]] = line["answer_eval"]

subquestions = {}
with open(subquestions_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        subquestions[line["data_id"]] = line["subquestions"]

contexts = {}
with open(reranked_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        current_context = []
        for i in range(len(line["sorted_context_nodes"])):
            if line["sorted_relevance_scores"][i] >= LIMIT:
                current_context.append((line["sorted_context_nodes"][i], line["sorted_relevance_scores"][i]))
        contexts[line["qid"]] = current_context

with open(reranked_path_sub, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        current_context = []
        for i in range(len(line["sorted_context_nodes"])):
            for j in range(len(line["sorted_context_nodes"][i])):
                if line["sorted_relevance_scores"][i][j] >= LIMIT:
                    current_context.append((line["sorted_context_nodes"][i][j], line["sorted_relevance_scores"][i][j]))
        contexts[line["qid"]].extend(current_context)

for qid in contexts.keys():
    contexts[qid] = sorted(contexts[qid], key=lambda x: x[1], reverse=True)
    current_context = []
    collected_context_nodes = set()
    for node_id, score in contexts[qid]:
        if node_id not in collected_context_nodes:
            current_context.append((node_id, score))
            collected_context_nodes.add(node_id)
    contexts[qid] = current_context

image_entity_mapping_path = os.path.join(BASE_PATH, "1-Preprocess", "data", "image_entity_mapping.jsonl")
image_entity_mapping = {}
with open(image_entity_mapping_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        image_path = line["image_file"]
        entities = "\n".join(line["entities"])
        image_entity_mapping[image_path] = entities

#load images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def extract_id(filename):
    return os.path.splitext(filename)[0]

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

images = {}
for file in os.listdir(oven_path):
    if is_image_file(file):
        images[extract_id(file)] = encode_image(os.path.join(oven_path, file))
    else:
        print(f"Invalid file: {file}")

#load_progress:
processed_questions = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            processed_questions.add(line["qid"])


#run loop:
MAX_ATTEMPTS = 30
with open(output_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        if qid in processed_questions:
            continue
        question = questions[qid][0]
        image, mime = images[questions[qid][1]]
        content = [{"type": "text", "text": "[CONTEXT]"}]
        for i in range(len(contexts[qid])):
            context_node_id, score = contexts[qid][i]
            if score >= LIMIT:
                content.append({"type": "text", "text": f"---Context {i+1}---"})
                context_node = nodes[context_node_id]
                if context_node.node_type == "V":
                    content.append({"type": "text", "text": f"This is an image of: {image_entity_mapping[os.path.basename(context_node.content)]}"})
                    context_image, context_image_mime = encode_image(context_node.content)
                    content.append({"type": "image_url", "image_url": {"url": f"data:{context_image_mime};base64,{context_image}"}})
                else:
                    content.append({"type": "text", "text": context_node.content})
        content.extend([
            {"type": "text", "text": f"[QUESTION]\n{question}"},
            {"type": "text", "text": "[QUESTION IMAGE]"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}}
        ])
        if COT:
            content.append({"type": "text", "text": "[PREREQUISITE QUESTIONS]."})
            content.append({"type": "text", "text": "\n".join(subquestions[qid])})
            system_prompt = cot_answer_prompt()
        else:
            system_prompt = answer_prompt()
        content.append({"type": "text", "text": f"[QUESTION]\n{question}"})
        for attempt in range(1,MAX_ATTEMPTS+1):
            response_text = None
            try:
                response_text, token = call_api(content=content, system_prompt=system_prompt, model="", mode="self-host")
                response = response_text.strip()
                if not isinstance(response, str) or not response:
                    raise ValueError("Invalid response")
                else:
                    processed_questions.add(qid)
                    data = {"qid": qid, "question": questions[qid][0], "image_id": questions[qid][1], "answer": response, "answer_eval":answer_eval[qid] , "token": token}
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f.flush()
                    break
            except Exception as e:
                print(f"Attempt {attempt} failed for question {qid}-{questions[qid][1]}: {e}")
                print(response_text)
                if attempt == MAX_ATTEMPTS:
                    print(f"Failed on question {qid}: {question}: {e}")
                    raise TimeoutError("A question failed")
                time.sleep(2)

print(f"Completed: {len(processed_questions)}/{len(questions)}")