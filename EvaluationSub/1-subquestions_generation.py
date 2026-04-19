import os
import json
import time
import ast
from tqdm import tqdm
from LLM.call_api import call_api
from LLM.prompts.subquestion_prompt import subquestion_prompt
import base64

#paths and load data
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
question_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "sampled_questions.jsonl"))
oven_path = os.path.normpath(os.path.join(BASE_PATH, "InfoSeek", "oven_images_sampled"))
subquestions_path = os.path.normpath(os.path.join(DIR_PATH, "data", "subquestions.jsonl"))

questions = {}
with open(question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        questions[line["data_id"]] = (line["question"],line["image_id"])

subquestions = set()
if os.path.exists(subquestions_path):
    with open(subquestions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            subquestions.add(line["qid"])

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
    for i in l:
        if not isinstance(i, str):
            return False
        if not i.strip():
            return False
        if i.strip()[-1] != "?":
            return False
    return True

#run loop:
MAX_ATTEMPTS = 30
with open(subquestions_path, "a", encoding="utf-8") as f:
    for qid in tqdm(questions.keys()):
        if qid in subquestions:
            continue
        question = questions[qid][0]
        image, mime = images[questions[qid][1]]
        prompt = subquestion_prompt(question)
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image}"}
            }
        ]
        for attempt in range(1,MAX_ATTEMPTS+1):
            try:
                response_text, token = call_api(content=content, model="", mode="self-host")
                response = ast.literal_eval(simple_strip(response_text))
                if not validate_list(response):
                    raise ValueError("Invalid response")
                else:
                    subquestions.add(qid)
                    data = {"qid": qid, "question": questions[qid][0], "image_id": questions[qid][1], "subquestions": response, "token": token}
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f.flush()
                    break
            except Exception as e:
                print(f"Attempt {attempt} failed for question {qid}-{questions[qid][1]}: {e}")
                print(response_text)
                if attempt == MAX_ATTEMPTS:
                    print(f"Failed on question {qid}: {question}: {e}")
                    raise TimeoutError("A question failed")
                time.sleep(5)

print(f"Completed: {len(subquestions)}/{len(questions)}")
