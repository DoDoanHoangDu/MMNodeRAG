import json
from sentence_transformers import SentenceTransformer
from LLM.call_api import call_api
from LLM.prompts.entity_matching_prompt import entity_matching_prompt
import torch
import faiss
import networkx as nx
import re
import ast
import time
import os
from tqdm import tqdm
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
decomposition_path = os.path.join(DIR_PATH, "data/decomposition.jsonl")

# -----------------------------
# 1. Entities Set
# -----------------------------
entities = set()
with open(decomposition_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        response = line['response']
        for s in response:
            for e in s['entities']:
                entities.add(e.strip().upper())

entities = list(entities)
print(f"Entities: {len(entities)}")

# -----------------------------
# 2. Embedding
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
def embed_entities(entities):
    inputs = [f"Entity: {e}" for e in entities]
    embeddings = model.encode(inputs, normalize_embeddings=True, show_progress_bar=True)
    return embeddings
print("Embedding entities:")
embeddings = embed_entities(entities)

#faiss
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"Embedding dimension: {dimension}")

# -----------------------------
# 3. Heuristic: Acronym Detection
# -----------------------------
def is_acronym(e1, e2):
    def normalize(e):
        return re.sub(r'[^A-Z0-9]', '', e.upper())

    def extract_initials(phrase):
        words = re.findall(r'[A-Z]+', phrase.upper())
        return ''.join(word[0] for word in words if word)

    e1_clean = normalize(e1)
    e2_clean = normalize(e2)

    e1_init = extract_initials(e1)
    e2_init = extract_initials(e2)

    # --- core check ---
    direct_match = (
        e1_clean == e2_clean or
        e1_clean == e2_init or
        e2_clean == e1_init
    )

    # --- guard against trivial matches ---
    if direct_match:
        # avoid 1-letter "acronyms"
        if min(len(e1_clean), len(e2_clean)) < 2:
            return False
        return True

    return False
# -----------------------------
# 4. Clustering
# -----------------------------
k = 50
D, I = index.search(embeddings, k)
embedding_similarity_graph = nx.Graph()
tau = 0.95

for i in range(len(embeddings)):
    for j, sim in zip(I[i], D[i]):
        if j == -1 or i == j:
            continue
        if sim >= tau or is_acronym(entities[i], entities[j]):
            embedding_similarity_graph.add_edge(i, j)

clusters = list(nx.find_cliques(embedding_similarity_graph))
clusters = [[int(node) for node in c] for c in clusters]
print(f"Total original clusters: {len(clusters)}")
print(f"Average cluster size: {sum(len(c) for c in clusters) / len(clusters):.2f}")
print(f"Max cluster size: {max(len(c) for c in clusters)}")

# -----------------------------
# 5. LLM-based Entity Resolution
# -----------------------------
def validate_response(response, original_entities):
    if not isinstance(response, list):
        raise ValueError("Response should be a list of clusters")
    for item in response:
        if not isinstance(item, list):
            raise ValueError("Each item in the response should be a list of entities")
        for entity in item:
            if not isinstance(entity, str):
                raise ValueError("Each entity in the response should be a string")
    flat_response = [e for cluster in response for e in cluster]
    if set(flat_response) != set(original_entities):
        raise ValueError("Response entities do not match original entities")

#construct base graph
synonym_graph = nx.Graph()
for entity in entities:
    synonym_graph.add_node(entity)
total_tokens = 0
for cluster in clusters:
    group = [entities[node] for node in cluster]
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            synonym_graph.add_edge(group[i], group[j])


#detect large connected components to validate with LLM
e_list = set()
for group in list(nx.connected_components(synonym_graph)):
    if len(group) > 5:
        e_list.update(group)

problematic_edges = [
    (u, v) for u, v in synonym_graph.edges()
    if u in e_list and v in e_list
]
synonym_graph.remove_edges_from(problematic_edges)

problematic_clusters = []
for cluster in clusters:
    if len(cluster) > 10:
        continue
    if any(entities[i] in e_list for i in cluster):
        problematic_clusters.append([entities[i] for i in cluster])

def merge_lists(lists):
    sets = [set(lst) for lst in lists]
    merged = True
    while merged:
        merged = False
        new_sets = []
        used = [False] * len(sets)

        for i in range(len(sets)):
            if used[i]:
                continue
            current = sets[i]
            for j in range(i + 1, len(sets)):
                if used[j]:
                    continue
                if len(set(current) & set(sets[j])) > 0 and len(current | sets[j]) <= 30:
                    current = current | sets[j]
                    used[j] = True
                    merged = True

            used[i] = True
            new_sets.append(current)
        sets = new_sets
    return [list(s) for s in sets]

problematic_clusters_merged = merge_lists(problematic_clusters)
problematic_clusters_merged = [cluster for cluster in problematic_clusters_merged if len(cluster) <= 15]

for cluster in tqdm(problematic_clusters_merged, desc="Processing clusters"):
    if len(cluster) > 1:
        system_prompt, user_prompt = entity_matching_prompt(cluster)
        MAX_ATTEMPTS = 10
        for attempt in range(MAX_ATTEMPTS):
            try:
                response, token = call_api(user_prompt, system_prompt=system_prompt, mode="self-host")
                response = ast.literal_eval(response.strip())
                validate_response(response, cluster)
                total_tokens += token
            except Exception as e:
                print(f"Attempt {attempt+1} failed for cluster {cluster}: {e}")
                time.sleep(2)
                if attempt == MAX_ATTEMPTS - 1:
                    print(f"Max attempts reached. Skipping cluster: {cluster}.")
                    response = [cluster]  # fallback to original cluster
        for group in response:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    synonym_graph.add_edge(group[i], group[j])

# Save synonym graph as edgelist
output_path = os.path.join(DIR_PATH, "data/synonym_graph.edgelist")
nx.write_edgelist(synonym_graph, output_path, delimiter="\t", data=False)
print(f"Total tokens used for entity resolution: {total_tokens}")
print(f"Total synonym edges: {synonym_graph.number_of_edges()}")





"""
synonym_graph = nx.Graph()
for entity in entities:
    synonym_graph.add_node(entity)
total_tokens = 0
for cluster in tqdm(clusters, desc="Processing clusters"):
    if len(cluster) > 1:
        prompt = entity_matching_prompt([entities[i] for i in cluster])
        MAX_ATTEMPTS = 20
        for attempt in range(MAX_ATTEMPTS):
            try:
                response,token = call_api(prompt)
                response = ast.literal_eval(response.strip())
                validate_response(response, [entities[i] for i in cluster])
                total_tokens += token
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed for cluster {cluster}: {e}")
                time.sleep(15* (attempt+1))
                if attempt == MAX_ATTEMPTS - 1:
                    print(f"Max attempts reached. Skipping cluster: {cluster}.")
                    response = [[entities[i] for i in cluster]]  # fallback to original cluster
        for group in response:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    synonym_graph.add_edge(group[i], group[j])

# Save synonym graph as edgelist
output_path = os.path.join(DIR_PATH, "data/synonym_graph.edgelist")
nx.write_edgelist(synonym_graph, output_path, delimiter="\t", data=False)
print(f"Total tokens used for entity resolution: {total_tokens}")
print(f"Total synonym edges: {synonym_graph.number_of_edges()}")
"""