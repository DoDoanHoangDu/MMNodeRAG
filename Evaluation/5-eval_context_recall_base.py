import os
import pickle
import json
from Metrics.context_recall import compute_context_recall
from tqdm import tqdm
import numpy as np
import re

#parameters
KNN = int(input("KNN: "))
REASONING = False

#paths
g4_path = os.path.join("2-Build_Graph/data/g4.pkl")
knn_path = os.path.join("Evaluation", "data", "knn.jsonl")
context_path = os.path.join("Evaluation", "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_reranked.jsonl")
question_path = os.path.normpath(os.path.join( "InfoSeek", "sampled_questions.jsonl"))
evaluated_base_path = os.path.join("Evaluation", "data", f"evaluation_{KNN}nn{"_reasoning" if REASONING else ""}_base.jsonl")
eval_base_path = os.path.join("Evaluation", "data", f"evaluation_context_recall_{KNN}nn{"_reasoning" if REASONING else ""}_base.jsonl")
print(question_path)
print(context_path)
print(evaluated_base_path)
print(eval_base_path)
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
        questions[line["data_id"]] = (line["question"], line["answer_eval"])

knns = {}
with open(knn_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        knns[line["qid"]] = line["knn"][:KNN]

contexts_base = {}
with open(context_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        current_context = [nid for nid in line["sorted_context_nodes"] if nid in knns[line["qid"]]]
        if len(current_context) != KNN:
            print(f"Warning: Question {line['qid']} has {len(current_context)} context nodes instead of {KNN}")
        contexts_base[line["qid"]] = current_context

#load evaluated
evaluated_base = {}
if os.path.exists(evaluated_base_path):
    with open(evaluated_base_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            score = int(line["score"])
            if score in {0,1}:
                evaluated_base[line["qid"]] = score

#load_progress:
processed_questions_base = {}
if os.path.exists(eval_base_path):
    with open(eval_base_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            value = line["context_recall"]
            if np.isnan(value):
                continue
            processed_questions_base[line["qid"]] = 1


with open(eval_base_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        context_recall, tokens = 0, 0

        if qid in processed_questions_base:
            continue
        if qid in evaluated_base and evaluated_base[qid] == 1:
            context_recall = 1
            tokens = 0
        else:
            question = questions[qid][0]
            answer_eval = questions[qid][1]
            answer_str = []
            for item in answer_eval:
                if isinstance(item, str):
                    answer_str.append(f"An acceptable answer is: {item}.")
                elif isinstance(item, dict):
                    answer_str.append(f"The exact answer is {item["value"]}. The answer lies in the range from {item["range"][0]} to {item["range"][1]}.")
            answer_str = " ".join(answer_str)

            context_nodes = contexts_base[qid]
            context_nodes_content = []
            for c in context_nodes:
                if nodes[c].node_type == "V":
                    continue
                content = nodes[c].content
                if any(isinstance(ans,str) and re.search(rf"\b{re.escape(ans)}\b", content, re.IGNORECASE) for ans in answer_eval):
                    context_recall = 1
                    tokens = 0
                context_nodes_content.append(nodes[c].content)

            if context_recall == 0:
                context_recall, tokens = compute_context_recall(
                    question=question,
                    contexts=context_nodes_content,
                    reference_answer=answer_str
                )
        data = {
            "qid": qid,
            "context_recall": 1 if context_recall > 0 else 0,
            "tokens": tokens
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()

total_recall = 0
total_tokens = 0
count = 0
records = []
with open(eval_base_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        if np.isnan(line["context_recall"]):
            print(f"Warning: context recall is NaN for question {line}")
            continue
        total_recall += line["context_recall"]
        total_tokens += line["tokens"]
        count += 1
        records.append(line)

print(f"Average Context Recall: {total_recall / count:.4f}")
print(f"Average Tokens: {total_tokens / count:.2f}")
print(f"Total Questions Evaluated: {count}")

with open(eval_base_path, "w", encoding="utf-8") as f:
    for obj in records:
        f.write(json.dumps(obj) + "\n")