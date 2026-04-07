import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import torch

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
image_dir = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "wikipedia_images_sampled"))
decomposition_path = os.path.join(DIR_PATH, "data/decomposition.jsonl")
knowledge_base_path = os.path.join(BASE_DIR, "InfoSeek", "KnowledgeBase.jsonl")

#Embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
def embed_entities(entities):
    inputs = [f"Entity: {e}" for e in entities]
    embeddings = model.encode(inputs, normalize_embeddings=True, show_progress_bar=True)
    return embeddings

#Main functions
def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def extract_id(filename):
    return os.path.splitext(filename)[0]

def get_relevant_document(document_id):
    relevant_document = None
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            if line["wikidata_id"] == document_id:
                relevant_document = line
                break
    return relevant_document

def get_relevant_entities(document_id):
    relevant_entities = set()
    with open(decomposition_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            doc_id = line["chunk_id"].split(":")[0]
            if doc_id == document_id:
                response = line['response']
                for s in response:
                    for e in s['entities']:
                        relevant_entities.add(e.strip().upper())
    return relevant_entities


image_entity_mapping = {}
for file in os.listdir(image_dir):
    if not is_image_file(file):
        continue
    file_id = extract_id(file)
    relevant_document = get_relevant_document(file_id)
    relevant_entities = get_relevant_entities(file_id)
    if relevant_document is None:
        continue
    title = relevant_document["wikipedia_title"].upper()
    if title in relevant_entities:
        image_entity_mapping[file] = [title]
    else:
        candidate_entities = [title] + list(relevant_entities)
        embeddings = embed_entities(candidate_entities)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        query = embeddings[0].reshape(1, -1)
        _, indices = index.search(query, 2)
        matched_index = indices[0][1]
        matched_entity = candidate_entities[matched_index]
        image_entity_mapping[file] = [matched_entity]
        print(f"Image: {file}, Title: {title}, Matched Entity: {matched_entity}")

image_mapping_path = os.path.join(DIR_PATH, "data/image_entity_mapping.jsonl")
with open(image_mapping_path, 'w', encoding='utf-8') as f:
    for image_file, entities in image_entity_mapping.items():
        json_line = json.dumps({"image_file": image_file, "entities": entities})
        f.write(json_line + '\n')
print(f"Total images processed: {len([k for k, v in image_entity_mapping.items() if v])}")
