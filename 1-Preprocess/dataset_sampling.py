import json
import random
import os
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)

entity_ids = set()
train_kb_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "infoseek_train_withkb.jsonl"))
val_kb_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "infoseek_val_withkb.jsonl"))
with open(train_kb_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        entity_ids.add(line["entity_id"])

with open(val_kb_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        entity_ids.add(line["entity_id"])


corpus_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "Wiki6M_ver_1_0.jsonl"))
knowledge_base_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "KnowledgeBase.jsonl"))
with open(corpus_path, 'r', encoding='utf-8') as f:
    original_count = sum(1 for line in f)

if os.path.exists(knowledge_base_path):
    os.remove(knowledge_base_path)

record_count = 0
sample_rate = 100000 / original_count
with open(corpus_path, 'r', encoding='utf-8') as f, \
    open(knowledge_base_path, 'a', encoding='utf-8') as kf:
    for line in f:
        line = json.loads(line)
        if line["wikidata_id"] in entity_ids or random.random() <= sample_rate:
            kf.write(json.dumps(line, ensure_ascii=False) + "\n")
            kf.flush()
            record_count += 1
        else:
            continue
print(f"Total records written to knowledge base: {record_count}")