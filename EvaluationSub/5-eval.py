import os
import json
from LLM.prompts.evaluation_prompt import evaluation_prompt
from LLM.call_api import call_api
from tqdm import tqdm
import time

#parameters
KNN = int(input("KNN: "))
REASONING = False

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
original_question_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "sampled_questions.jsonl"))
output_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"evaluation_{KNN}nn.jsonl"))
input("Confirm?")

#load data
original_questions = {}
with open(original_question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        original_questions[line["data_id"]] = line["question"]

questions = {}
for level in range(100):
    answers_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"answers_{KNN}nn_{level}.jsonl"))
    if not os.path.exists(answers_path):
        break
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
            questions[line["qid"]] = (original_questions[line["qid"]],line["answer"],answer_eval)

#load_progress:
processed_questions = {}
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            processed_questions[line["qid"]] = int(line["score"])

#run loop:
MAX_ATTEMPTS = 30
with open(output_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        if qid in processed_questions:
            continue
        question,answer,answer_eval = questions[qid]
        evaluation_input = {
            "question": question,
            "model_answer": answer,
            "acceptable_answers": answer_eval
        }
        content = [
            {"type": "text", "text": f"[USER INPUT]\n{json.dumps(evaluation_input)}"},
        ]

        for attempt in range(1,MAX_ATTEMPTS+1):
            try:
                response_text, token = call_api(content=content, system_prompt=evaluation_prompt(), model="", mode="self-host")
                response = int(response_text.strip())
                if not isinstance(response, int) or response not in [0, 1]:
                    raise ValueError("Invalid response")
                else:
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