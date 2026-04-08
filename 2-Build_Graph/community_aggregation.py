from Node import Node
import pickle
import os
import json
import ast
from tqdm import tqdm
import igraph as ig
from leidenalg import find_partition, ModularityVertexPartition
from collections import defaultdict
from LLM.call_api import call_api
from LLM.prompts.high_level_elements_prompt import high_level_elements_prompt
from LLM.prompts.high_level_overview_prompt import high_level_overview_prompt
import time


#load data
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g2_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g1.pkl")

with open(g2_path, "rb") as f:
    nodes = pickle.load(f)

#leiden commnunity detection
def leiden_community_detection(node_dict):
    id_to_idx = {nid: i for i, nid in enumerate(node_dict.keys())}
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    edge_weights = defaultdict(float)
    for nid, node in node_dict.items():
        for neighbor, weight in node.edges.items():
            a, b = id_to_idx[nid], id_to_idx[neighbor]
            if a < b:
                edge_weights[(a, b)] += weight

    edges = list(edge_weights.keys())
    weights = list(edge_weights.values())

    g = ig.Graph(n=len(node_dict), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = find_partition(g, ModularityVertexPartition, weights="weight")
    membership = partition.membership
    communities = {idx_to_id[i] : membership[i] for i in range(len(node_dict))}
    return communities

node_communities = leiden_community_detection(nodes)
communities = defaultdict(list)
for node,community in node_communities.items():
    community_id = f'H{community:06d}'
    communities[community_id].append(node)
print(f"Number of communities detected: {communities}")

#get LLM summaries
def format_list(l):
    ans = []
    for i in range(len(l)):
        ans.append(f"[{i+1}] {l[i]}")
    return "\n".join(ans)

community_summaries = {}
for community_id in tqdm(communities):
    if community_id in community_summaries:
        continue
    semantic_count = 0
    nodes_id = communities[community_id]
    content = []
    for nid in nodes_id:
        node = nodes[nid]
        if node.node_type == "S":
            semantic_count+=1
        if node.node_type in ["R", "N", "V"]:
            continue
        content.append(node.content)
    if semantic_count < 2:
        continue
    else:
        content = format_list(content)
        prompt = high_level_elements_prompt(content)
        MAX_ATTEMPTS = 30
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response, token = call_api(prompt, model="", mode="gemini")
                community_summaries[community_id] = (response, token)
                time.sleep(3)
            except Exception as e:
                print(f"Attempt {attempt} failed for community {community_id}: {e}")
                if attempt == MAX_ATTEMPTS:
                    print(f"Failed on community {community_id} with context length {len(prompt.split())}: {e}")
                    raise TimeoutError("A community failed")
                time.sleep(5 * attempt)
        
print(f"Number of communities summarized: {len(community_summaries)}")

#get llm overviews
def validate_overview(overview):
    if not isinstance(overview, list):
        return False
    for o in overview:
        if not isinstance(o, str):
            return False
    return True

community_overviews = {}
for community_id in tqdm(community_summaries.keys()):
    if community_id in community_overviews:
        continue
    summary = community_summaries[community_id][0]
    prompt = high_level_overview_prompt(summary)
    MAX_ATTEMPTS = 30
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            overview, token =  call_api(prompt, model="", mode="gemini")
            overview = ast.literal_eval(overview)
            if not validate_overview(overview):
                raise ValueError("Invalid overview format")
            community_overviews[community_id] = (overview, token)
            time.sleep(3)
        except Exception as e:
            print(f"Attempt {attempt} failed for community {community_id}: {e}")
            if attempt == MAX_ATTEMPTS:
                print(f"Failed on community {community_id} with context length {len(prompt.split())}: {e}")
                raise TimeoutError("A community failed")
            time.sleep(5 * attempt)

print(f"Number of community overviews: {len(community_overviews)}")

#save
communities_path = os.path.join(DIR_PATH, "data", "communities.jsonl")
with open(communities_path, "w", encoding="utf-8") as f:
    for community_id in community_summaries.keys():
        members = communities[community_id]
        summary, token_summary = community_summaries[community_id]
        overview, token_overview = community_overviews[community_id]
        data = {
            "community_id": community_id, 
            "members": members,
            "summary": summary,
            "overview": overview,
            "token_summary": token_summary,
            "token_overview": token_overview     
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()