import base64
import os
import json
from LLM.prompts.evaluation_prompt import evaluation_prompt
from LLM.call_api import call_api
from tqdm import tqdm
import time
from Metrics.fuzzy_accuracy import fuzzy_accuracy
import pickle

#parameters
KNN = int(input("KNN: "))
REASONING = False
BASE = True if input("BASE (y/n): ").lower() == "y" else False 
LIMIT = 0.5

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
knn_path = os.path.join(BASE_PATH, "Evaluation/data/knn.jsonl")
reranked_path = os.path.join(DIR_PATH, "data", f"context_{KNN if KNN > 0 else 8}nn{"_reasoning" if REASONING else ""}_reranked.jsonl")
answers_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"answers_{KNN}nn{'_base' if BASE else ''}.jsonl"))
output_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"evaluation_{KNN}nn{'_base' if BASE else ''}.jsonl"))
print(answers_path)
input("Confirm?")

#load questions
questions = {}
with open(answers_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        answer_eval = line["answer_eval"]
        if not isinstance(answer_eval, list):
            print(f"Invalid answer_eval for question {line['qid']}: {answer_eval}")
            raise ValueError("Invalid answer_eval format")
        for i in range(len(answer_eval)):
            val = answer_eval[i]
            if not (isinstance(val, (str, int, float)) or (isinstance(val, dict) and "wikidata" in val and "range" in val)):
                print(f"Invalid answer_eval value for question {line['qid']}: {val}")
                raise ValueError("Invalid answer_eval value format")
            if isinstance(val, dict):
                answer_eval[i]["value"] = answer_eval[i].pop("wikidata")
        questions[line["qid"]] = (line["question"],line["answer"],answer_eval)

#load knn and contexts
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

def valid_context(context_node_id, relevance_score, knns = None):
    if relevance_score < LIMIT:
        return False
    if BASE:
        return context_node_id in knns
    return True

knns = {}
with open(knn_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        knns[line["qid"]] = line["knn"][:KNN]

contexts = {}
with open(reranked_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        current_context = []
        for i in range(len(line["sorted_context_nodes"])):
            if valid_context(line["sorted_context_nodes"][i], line["sorted_relevance_scores"][i], knns[line["qid"]]):
                current_context.append(line["sorted_context_nodes"][i])
        contexts[line["qid"]] = current_context

image_entity_mapping_path = os.path.join(BASE_PATH, "1-Preprocess", "data", "image_entity_mapping.jsonl")
image_entity_mapping = {}
with open(image_entity_mapping_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        image_path = line["image_file"]
        entities = "\n".join(line["entities"])
        image_entity_mapping[image_path] = entities

#load_progress:
processed_questions = {}
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            processed_questions[line["qid"]] = int(line["score"])

#run loop:
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

MAX_ATTEMPTS = 30
with open(output_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        if qid in processed_questions:
            continue
        question,answer,answer_eval = questions[qid]
        fuzzy_score = fuzzy_accuracy(answer, answer_eval)
        if fuzzy_score == 1:
            processed_questions[qid] = 1
            data = {"qid": qid, "score": 1, "token": 0}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()
            continue

        content = []
        if len(contexts[qid]) > 0:
            content.append({"type": "text", "text": "[CONTEXT]"})
        for i in range(len(contexts[qid])):
            context_node_id = contexts[qid][i]
            content.append({"type": "text", "text": f"---Context {i+1}---"})
            context_node = nodes[context_node_id]
            if context_node.node_type == "V":
                content.append({"type": "text", "text": f"This is an image of: {image_entity_mapping[os.path.basename(context_node.content)]}"})
                context_image, context_image_mime = encode_image(context_node.content)
                content.append({"type": "image_url", "image_url": {"url": f"data:{context_image_mime};base64,{context_image}"}})
            else:
                content.append({"type": "text", "text": context_node.content})

        evaluation_input = {
            "question": question,
            "model_answer": answer,
            "acceptable_answers": answer_eval
        }
        content.append({"type": "text", "text": f"[USER INPUT]\n{json.dumps(evaluation_input, indent=2, ensure_ascii=False)}"})

        for attempt in range(1,MAX_ATTEMPTS+1):
            response_text = None
            try:
                for _ in range(2):
                    response_text, token = call_api(content=content, system_prompt=evaluation_prompt(), model="", mode="self-host")
                    response = int(response_text.strip())
                    if not isinstance(response, int) or response not in [0, 1]:
                        raise ValueError("Invalid response")
                    if response == 1:
                        break
            
                processed_questions[qid] = response
                data = {"qid": qid, "score": response, "token": token}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
                processed_questions[qid] = response
                break
            except Exception as e:
                print(f"Attempt {attempt} failed for question {qid}-{questions[qid][1]}: {e}")
                print(response_text)
                if attempt == MAX_ATTEMPTS:
                    print(f"Failed on question {qid}: {question}: {e}")
                    raise TimeoutError("A question failed")
                time.sleep(5)

print(f"Completed: {len(processed_questions)}/{len(questions)}")
print(f"Total score: {sum(processed_questions.values())}/{len(questions)}")