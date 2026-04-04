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


#Write sampled knowledge base
corpus_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "Wiki6M_ver_1_0.jsonl"))
knowledge_base_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "KnowledgeBase.jsonl"))
with open(corpus_path, 'r', encoding='utf-8') as f:
    original_count = sum(1 for line in f)

if os.path.exists(knowledge_base_path):
    os.remove(knowledge_base_path)

record_count = 0
word_count = 0
sample_rate = 0 * len(entity_ids) / original_count
with open(corpus_path, 'r', encoding='utf-8') as f, \
    open(knowledge_base_path, 'a', encoding='utf-8') as kf:
    for line in f:
        line = json.loads(line)
        if line["wikidata_id"] in entity_ids or random.random() <= sample_rate:
            kf.write(json.dumps(line, ensure_ascii=False) + "\n")
            kf.flush()
            record_count += 1
            word_count += len(line["wikipedia_content"].split())
        else:
            continue


print(f"Total records written to knowledge base: {record_count}")
print(f"Total words in knowledge base: {word_count}")