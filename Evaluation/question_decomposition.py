import os
import json
import time
import ast
from tqdm import tqdm
from LLM.call_api import call_api
from LLM.prompts.question_decompose_prompt import question_decompose_prompt

#paths and load data
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
question_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "sampled_questions.jsonl"))
decompoistion_path = os.path.normpath(os.path.join(DIR_PATH, "data", "decomposed_questions.jsonl"))

questions = {}
with open(question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        questions[line["data_id"]] = line["question"]

decompositions = {}
if os.path.exists(decompoistion_path):
    with open(decompoistion_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            questions[line["data_id"]] = line["entities"]

#validate llm list
def simple_strip(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:] # Remove ```json
    elif text.startswith("```"):
        text = text[3:] # Remove ```
    
    if text.endswith("```"):
        text = text[:-3] # Remove closing ```
    return text.strip()

def validate_list(l):
    if not isinstance(l, list):
        return False
    if not len(l) > 0:
        return False
    for i in l:
        if not isinstance(i, str):
            return False
        if not i.strip():
            return False
    return True

#run loop:
MAX_ATTEMPTS = 30
with open(decompoistion_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        if qid in decompositions:
            continue
        question = questions[qid]
        prompt = question_decompose_prompt(question)
        for attempt in range(1,MAX_ATTEMPTS+1):
            try:
                response_text, token = call_api(prompt, model="", mode="self-host")
                response = ast.literal_eval(simple_strip(response_text))
                if not validate_list(response):
                    raise ValueError("Invalid response")
                decompositions[qid] = response
                data = {"data_id": qid, "entities": response, "token": token}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"Attempt {attempt} failed for question {qid}: {e}")
                print(response_text)
                if attempt == MAX_ATTEMPTS:
                    print(f"Failed on question {qid}: {question}: {e}")
                    raise TimeoutError("A question failed")
                time.sleep(5 * attempt)

print(f"Completed: {len(decompositions)}/{len(questions)}")
