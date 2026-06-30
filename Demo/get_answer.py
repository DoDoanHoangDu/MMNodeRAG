import os
import json
from LLM.prompts.answer_prompt import answer_prompt
from LLM.call_api import call_api
import base64
import time

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

image_entity_mapping = {}
with open("1-Preprocess/data/image_entity_mapping.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        image_path = line["image_file"]
        entities = "\n".join(line["entities"])
        image_entity_mapping[image_path] = entities

#run loop:
def get_answer(nodes, question, image_path, contexts, MAX_ATTEMPTS = 10):
    image, mime = encode_image(image_path)
    content = [
        {"type": "text", "text": f"[QUESTION]\n{question}"},
        {"type": "text", "text": "[QUESTION IMAGE]"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}},
        {"type": "text", "text": "[CONTEXT]"},
    ]
    for i in range(len(contexts)):
        context_node_id, score = contexts[i]
        if score < 0.5:
            continue
        content.append({"type": "text", "text": f"---Context {i+1}---"})
        context_node = nodes[context_node_id]
        if context_node.node_type == "V":
            content.append({"type": "text", "text": f"This is an image of: {image_entity_mapping[os.path.basename(context_node.content)]}"})
            context_image, context_image_mime = encode_image(context_node.content)
            content.append({"type": "image_url", "image_url": {"url": f"data:{context_image_mime};base64,{context_image}"}})
        else:
            content.append({"type": "text", "text": context_node.content})
    for attempt in range(1,MAX_ATTEMPTS+1):
        response_text = None
        try:
            response_text, token = call_api(content=content, system_prompt=answer_prompt(), model="", mode="self-host")
            response = response_text.strip()
            if not isinstance(response, str) or not response:
                raise ValueError("Invalid response")
            else:
                return response, token
        except Exception as e:
            print(f"Answer attempt {attempt} failed: {e}")
            print(response_text)
            time.sleep(1)
    return None, token
