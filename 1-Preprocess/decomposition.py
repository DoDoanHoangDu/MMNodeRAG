from LLM.call_api import call_api
from LLM.prompts.text_decomposition_prompt import text_decomposition_prompt
import os
import json
import time
from tqdm import tqdm
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)
 
#Response validation
def extract_json(text):
    text = text.strip()
    # remove markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
    return json.loads(text)
 
def is_valid_schema(data):
    if not isinstance(data, list):
        return False
 
    for item in data:
        if not isinstance(item, dict):
            return False
 
        # Required keys
        required_keys = {"semantic_unit", "entities", "relationships"}
        if not required_keys.issubset(item.keys()):
            return False
 
        #check empty
        if not item["semantic_unit"].strip():
            return False

        if not item["entities"]:
            return False

        if not item["relationships"]:
            return False
 
        # semantic_unit: string
        if not isinstance(item["semantic_unit"], str):
            return False
 
        # entities: list of strings
        if not isinstance(item["entities"], list) or \
            not all(isinstance(x, str) for x in item["entities"]):
            return False
 
        # relationships: list of strings
        if not isinstance(item["relationships"], list) or \
            not all(isinstance(x, str) for x in item["relationships"]):
            return False
    return True
 
#Check for processed chunk ids
ids_path = os.path.join(DIR_PATH, "data/processed_chunk_ids.txt")
processed_ids = set()
try:
    with open(ids_path, "r") as f:
        for line in f:
            processed_ids.add(line.strip())
except FileNotFoundError:
	pass
 
#Decompose each chunk
failed_count = 0
chunks_path = os.path.join(DIR_PATH, "data/chunks.jsonl")
output_path = os.path.join(DIR_PATH, "data/decomposition.jsonl")
with open(output_path, "a", encoding="utf-8") as outfile, \
    open(chunks_path, "r", encoding="utf-8") as chunks, \
    open(ids_path, "a") as idfile:
 
    def write(data):
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        outfile.flush()

    for line in tqdm(chunks, desc="Processing chunks"):
        record = json.loads(line)
        rid = str(record["chunk_id"])
        if rid in processed_ids:
            continue

        content = record.get("chunk_content", "")
        if not content:
            print(f"Empty content for chunk {rid}, skipping.")
            continue

        prompt = text_decomposition_prompt(content)
        max_retries = 10
        success = False
        for attempt in range(max_retries):
            try:
                response, token = call_api(prompt)
                response = extract_json(response)
                if not is_valid_schema(response):
                    raise ValueError("Invalid schema")

                data = {
                    "chunk_id": rid,
                    "response": response,
                    "token": token
                }
                write(data)
                success = True
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed on chunk {rid}: {e}")
                    failed_count += 1
                else:
                    time.sleep(10 * (attempt + 1))  # wait before next attempt
                
                continue
 
        if success:
            idfile.write(rid + "\n")
            idfile.flush()
            processed_ids.add(rid)
print("Failed count: ",failed_count)
 