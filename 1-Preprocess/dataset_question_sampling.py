import json
import random
import os
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)

#relevant entity ids
entity_ids = set()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
image_dir = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "wikipedia_images_sampled"))

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def extract_id(filename):
    return os.path.splitext(filename)[0]

for file in os.listdir(image_dir):
    if not is_image_file(file):
        continue

    file_id = extract_id(file)
    entity_ids.add(file_id)

questions = {eid: [] for eid in entity_ids}
train_q_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "infoseek_train_withkb.jsonl"))
val_q_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "infoseek_val_withkb.jsonl"))
with open(train_q_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        if line["entity_id"] in entity_ids:
            questions[line["entity_id"]].append(line["data_id"])

with open(val_q_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        if line["entity_id"] in entity_ids:
            questions[line["entity_id"]].append(line["data_id"])

sampled_question = { eid: random.sample(qids, min(2, len(qids))) for eid, qids in questions.items()}
sampled_question_set = set()
for qids in sampled_question.values():
    sampled_question_set.update(qids)
train_questions_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "infoseek_train.jsonl"))
val_questions_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "infoseek_val.jsonl"))
output = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "sampled_questions.jsonl"))
with open(train_questions_path, 'r', encoding='utf-8') as f, \
    open(val_questions_path, 'r', encoding='utf-8') as vf, \
    open(output, 'w', encoding='utf-8') as of:
    for line in f:
        line = json.loads(line)
        if line["data_id"] in sampled_question_set:
            of.write(json.dumps(line, ensure_ascii=False) + "\n")
            of.flush()
    for line in vf:
        line = json.loads(line)
        if line["data_id"] in sampled_question_set:
            of.write(json.dumps(line, ensure_ascii=False) + "\n")
            of.flush()

print(f"Sampled {len(sampled_question_set)} questions for {len(sampled_question)} entities.")