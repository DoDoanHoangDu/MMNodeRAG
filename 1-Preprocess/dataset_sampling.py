import json
import os
from tqdm import tqdm
import urllib.request
from PIL import Image
import time

KEEP = 10**4

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)
processed_wikis_path = os.path.join(DIR_PATH, "data", "processed_wikis.jsonl")
wikidata_path = os.path.join(BASE_DIR, "Dataset", "Wiki6M_ver_1_0.jsonl")
knowledge_base_path = os.path.join(BASE_DIR, "Dataset", "knowledge_base.jsonl")
wiki_images_path = os.path.join(BASE_DIR, "Dataset", "wiki_images")

processed_wikis = []
with open(processed_wikis_path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        record = json.loads(line)
        wiki_id = record["id"]
        processed_wikis.append((wiki_id, record["score"]))

print(f"Total processed wikis: {len(processed_wikis)}")
print("Sorting wikis by score...")
processed_wikis.sort(key=lambda x: x[1], reverse=True)
processed_wikis = processed_wikis[:KEEP]
processed_wikis = set(w[0] for w in processed_wikis)

records = []
with open(wikidata_path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        record = json.loads(line)
        wiki_id = record["wikidata_id"]
        if wiki_id in processed_wikis:
            records.append(record)

with open(knowledge_base_path, "w", encoding="utf-8") as f:
    for record in tqdm(records):
        wiki_id = record["wikidata_id"]
        wiki_image_url = record.get("wikipedia_image_url", "")
        if wiki_image_url:
            try:
                image_path = os.path.join(wiki_images_path, f"{wiki_id}.jpg")
                urllib.request.urlretrieve(wiki_image_url, image_path)
                record["image_path"] = image_path
                time.sleep(1)
            except Exception as e:
                print(f"Error occurred while downloading image for wiki_id {wiki_id}: {e}")
        f.write(json.dumps(record, ensure_ascii=False) + "\n")