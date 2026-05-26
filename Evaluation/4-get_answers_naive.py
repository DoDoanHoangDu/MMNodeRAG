import os
import pickle
import json
from LLM.prompts.answer_prompt import answer_prompt
from LLM.call_api import call_api
from tqdm import tqdm
import base64
import time
import pandas as pd
from PIL import Image
import ast
import io
import re

#parameters
KNN = int(input("KNN: "))
REASONING = False

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
questions0 = pd.read_parquet("Dataset/MMMU-Pro/test-00000-of-00002.parquet")
questions1 = pd.read_parquet("Dataset/MMMU-Pro/test-00001-of-00002.parquet")
questions = pd.concat([questions0, questions1], ignore_index=True)
output_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"answers_{KNN}nn_naive.jsonl"))
print(output_path)
input("Confirm?")

#load data
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)

#load_progress:
processed_questions = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            processed_questions.add(line["qid"])

#run loop:
def resize_image(image, max_size = 600):
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    scale = max_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.BILINEAR)

MAX_ATTEMPTS = 10
with open(output_path, "a", encoding="utf-8") as f:
    for idx, row in tqdm(questions.iterrows(), total=len(questions)):
        qid = row["id"]
        if qid in processed_questions:
            continue
        question = row["question"]

        options = ast.literal_eval(row["options"])
        options_text = "Pick only one correct answer:"
        choices = set()
        for i, item in enumerate(options):
            label = chr(ord('A') + i)
            options_text += "\n" + f"{label}) {item}"
            choices.add(label)
        
        images = []
        for i in range(1,8):
            img_dict = row[f"image_{i}"]
            if not img_dict or not isinstance(img_dict, dict):
                continue
            img_data = img_dict["bytes"]
            image = Image.open(io.BytesIO(img_data)).convert("RGBA").convert("RGB")
            image = resize_image(image)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images.append(image)

        content = [{"type": "text", "text": f"[QUESTION]\n{question}"}]
        for index, image in enumerate(images, start=1):
            content.append({"type": "text", "text": f"[IMAGE {index}]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}})
        content.append({"type": "text", "text": f"[OPTIONS]\n{options_text}"})

        for attempt in range(1,MAX_ATTEMPTS+1):
            try_content = content[:max(1 + 2 * len(images), len(content) - attempt + 1)]
            response_text = None
            try:
                response_text, token = call_api(content=try_content, system_prompt=answer_prompt(), model="", mode="self-host")
                match = re.search(r"The answer is ([A-Z])", response_text, re.IGNORECASE)
                response = match.group(1).upper()
                if not isinstance(response, str) or response not in choices:
                    raise ValueError("Invalid response")
                else:
                    processed_questions.add(qid)
                    data = {"qid": qid, "question": question, "answer": response, "token": token}
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f.flush()
                    break
            except Exception as e:
                print(f"Attempt {attempt} failed for question {qid}: {e}")
                print(response_text)
                if attempt == MAX_ATTEMPTS:
                    print(f"Failed on question {qid}: {question + "\n" + options_text}: {e}")
                    raise TimeoutError("A question failed")
                time.sleep(5)

print(f"Completed: {len(processed_questions)}/{len(questions)}")