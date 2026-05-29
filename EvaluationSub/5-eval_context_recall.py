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
LIMIT = 0.5

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
reranked_path_sub = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_reranked.jsonl")
reranked_path = os.path.join(BASE_PATH, "Evaluation", "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_reranked.jsonl")
question_path = os.path.normpath(os.path.join("InfoSeek", "sampled_questions.jsonl"))
evaluated_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"evaluation_{KNN}nn.jsonl"))
evaluated_path_COT = os.path.normpath(os.path.join(DIR_PATH, "data", f"evaluation_{KNN}nn_COT.jsonl"))
eval_path = os.path.join("Evaluation", "data", f"evaluation_context_recall_{KNN}nn{"_reasoning" if REASONING else ""}.jsonl")
eval_sub_path = os.path.join("EvaluationSub", "data", f"evaluation_context_recall_{KNN}nn.jsonl")
print(question_path)
print(evaluated_path)
print(eval_sub_path)
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
            current_context.append(node_id)
            collected_context_nodes.add(node_id)
    contexts[qid] = current_context

#load evaluated
evaluated = {}
if os.path.exists(evaluated_path):
    with open(evaluated_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            score = int(line["score"])
            if score in {0,1}:
                evaluated[line["qid"]] = score

evaluated_COT = {}
if os.path.exists(evaluated_path_COT):
    with open(evaluated_path_COT, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            score = int(line["score"])
            if score in {0,1}:
                evaluated_COT[line["qid"]] = score

print("Loaded evaluated questions:", sum(evaluated.values()))

#load_progress:
processed_questions_regular = {}
if os.path.exists(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            value = line["context_recall"]
            if np.isnan(value):
                continue
            processed_questions_regular[line["qid"]] = value
print("Loaded regular evaluated questions:", sum(processed_questions_regular.values()))

processed_questions = {}
if os.path.exists(eval_sub_path):
    with open(eval_sub_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            value = line["context_recall"]
            if np.isnan(value) or value == 0:
                continue
            processed_questions[line["qid"]] = 1
print("Loaded processed questions:", sum(processed_questions.values()))

with open(eval_sub_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        context_recall, tokens = 0, 0

        if qid in processed_questions:
            continue
        if qid in evaluated and evaluated[qid] == 1:
            context_recall = 1
            tokens = 0
        elif qid in evaluated_COT and evaluated_COT[qid] == 1:
            context_recall = 1
            tokens = 0
        elif qid in processed_questions_regular and processed_questions_regular[qid] == 1:
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
                    answer_str.append(f"The exact answer is {item["value"]}. Any answer lying in the range from {item["range"][0]} to {item["range"][1]} is acceptable.")
            answer_str = "\n".join(answer_str)

            context_nodes = contexts[qid]
            context_nodes_content = []
            for c in context_nodes:
                if nodes[c].node_type == "V":
                    continue
                content = nodes[c].content
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
with open(eval_sub_path, "r", encoding="utf-8") as f:
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

with open(eval_sub_path, "w", encoding="utf-8") as f:
    for obj in records:
        f.write(json.dumps(obj) + "\n")