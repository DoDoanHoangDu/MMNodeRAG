import json
import os
import shutil
import random
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)

#Valid ids
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

#Check image validity
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
def extract_id(filename):
    return os.path.splitext(filename)[0]

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

#Collect images and write to target directory
TARGET_COUNT = 500
image_count = 0
source_image_dir = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "wikipedia_images_full"))
target_image_dir = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "wikipedia_images_sampled"))
if os.path.exists(target_image_dir):
    shutil.rmtree(target_image_dir)
os.makedirs(target_image_dir, exist_ok=True)

#collect all images paths
candidates = []
for root, _, files in os.walk(source_image_dir):
    for file in files:
        if not is_image_file(file):
            continue

        file_id = extract_id(file)

        if file_id in entity_ids:
            full_path = os.path.join(root, file)
            candidates.append(full_path)
print(f"Total candidate images: {len(candidates)}")

#sample and copy
sampled = random.sample(candidates, TARGET_COUNT)
for src_path in sampled:
    filename = os.path.basename(src_path)
    dst_path = os.path.join(target_image_dir, filename)

    shutil.copy2(src_path, dst_path)

print(f"Exactly {TARGET_COUNT} images written.")