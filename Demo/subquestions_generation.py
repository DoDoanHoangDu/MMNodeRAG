import os
import time
import ast
from LLM.prompts.subquestion_prompt import subquestion_prompt
import base64
from LLM.call_api import call_api

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
def subquestion_generation(question, image_path, MAX_ATTEMPTS = 10):
    image, mime = encode_image(image_path)
    prompt = subquestion_prompt(question)
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image}"}
        }
    ]
    for attempt in range(1,MAX_ATTEMPTS+1):
        response_text = None
        try:
            response_text, token = call_api(content=content, model="", mode="self-host")
            response = ast.literal_eval(simple_strip(response_text))
            if not validate_list(response):
                raise ValueError("Invalid response")
            else:
                return response, token
        except Exception as e:
            print(f"Subquestion attempt {attempt} failed: {e}")
            print(response_text)
            time.sleep(1)
    return None, token
