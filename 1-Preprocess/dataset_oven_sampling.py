import os
import json
import shutil
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def shard_path(index):
    return f"{BASE_DIR}/InfoSeek/oven_images_full/{index:02d}"

def extract_id(filename):
    return os.path.splitext(filename)[0]

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

required_ids = set()
question_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "sampled_questions.jsonl"))
with open(question_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        required_ids.add(line["image_id"])

candidates = []
for i in range(1,100):
    source_path = shard_path(i)
    if os.path.exists(source_path):
        print(source_path)
        for root, _, files in os.walk(source_path):
            for file in files:
                if not is_image_file(file):
                    continue

                file_id = extract_id(file)

                if file_id in required_ids:
                    full_path = os.path.join(root, file)
                    candidates.append(full_path)

print(f"Processing: {len(candidates)}/{len(required_ids)}")

#copy
output_path = f"{BASE_DIR}/InfoSeek/oven_images_sampled"
os.makedirs(output_path, exist_ok=True)
for src_path in candidates:
    filename = os.path.basename(src_path)
    dst_path = os.path.join(output_path, filename)
    shutil.copy2(src_path, dst_path)

#check
img_count = 0
for root, _, files in os.walk(output_path):
    for file in files:
        if not is_image_file(file):
            raise ValueError("Invalid file in target")
        file_id = extract_id(file)

        if file_id not in required_ids:
            raise ValueError("Wrong image file copied")
        img_count += 1
print(f"Progress: {img_count}/{len(required_ids)}")