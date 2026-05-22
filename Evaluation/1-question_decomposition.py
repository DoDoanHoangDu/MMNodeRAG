import os
import json
import time
import ast
from tqdm import tqdm
from LLM.call_api import call_api
from LLM.prompts.question_decompose_prompt import question_decompose_prompt
import base64
import pandas as pd
from PIL import Image
import io

#paths and load data
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
questions0 = pd.read_parquet("Dataset/MMMU-Pro/test-00000-of-00002.parquet")
questions1 = pd.read_parquet("Dataset/MMMU-Pro/test-00001-of-00002.parquet")
questions = pd.concat([questions0, questions1], ignore_index=True)
decompoistion_path = os.path.normpath(os.path.join(DIR_PATH, "data", "decomposed_questions.jsonl"))

decompositions = {}
if os.path.exists(decompoistion_path):
    with open(decompoistion_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            decompositions[line["data_id"]] = line["entities"]

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
    return True

#run loop
def resize_image(image, max_size = 1000):
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    scale = max_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.LANCZOS)


MAX_ATTEMPTS = 30
with open(decompoistion_path, "a", encoding="utf-8") as f:
    for idx, row in tqdm(questions.iterrows()):
        qid = row["id"]
        if qid in decompositions:
            continue
        question = row["question"]
        options = ast.literal_eval(row["options"])
        for i, item in enumerate(options):
            label = chr(ord('A') + i)
            question += "\n" + f"{label}) {item}"
        images = []
        for i in range(1,8):
            img_dict = row[f"image_{i}"]
            if not img_dict or not isinstance(img_dict, dict):
                continue
            img_data = img_dict["bytes"]
            image = Image.open(io.BytesIO(img_data)).convert("RGBA").convert("RGB")
            image = resize_image(image)
            image = base64.b64encode(io.BytesIO(image.tobytes()).getvalue()).decode("utf-8")
            images.append(image)

        system_prompt, user_prompt = question_decompose_prompt(question)
        content = [{"type": "text", "text": user_prompt}]
        for image in images:
            image_content = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
            content.append(image_content)
        for attempt in range(1,MAX_ATTEMPTS+1):
            try:
                response_text, token = call_api(content=content, system_prompt= system_prompt, model="", mode="self-host")
                response = ast.literal_eval(simple_strip(response_text))
                if not validate_list(response):
                    raise ValueError("Invalid response")
                else:
                    decompositions[qid] = response
                    data = {"data_id": qid, "question": questions[qid][0], "image_id": questions[qid][1], "entities": response, "token": token}
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

print(f"Completed: {len(decompositions)}/{len(questions)}")
