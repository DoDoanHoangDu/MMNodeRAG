import os
import pickle
from Node import Node
import json
from tqdm import tqdm
from Retrieval.ppr_local import shallow_ppr_local
from Retrieval.shortest_path import all_pairs_shortest_paths

#parameters
QUESTION_LEVEL = input("Question Level: ")
KNN = input("Knn: ")
REASONING = False

#paths
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(DIR_PATH)
g4_path = os.path.join(BASE_PATH, "2-Build_Graph/data/g4.pkl")
knn_path = os.path.join(BASE_PATH, "Evaluation/data/knn.jsonl")
decompoistion_path = os.path.normpath(os.path.join(DIR_PATH, "data", f"decomposed_questions_{QUESTION_LEVEL}.jsonl"))
entities_path = os.path.join(BASE_PATH, "2-Build_Graph/data", "entities.jsonl")
relevant_nodes_path = os.path.join(DIR_PATH, "data", f"context_{KNN}nn{"_reasoning" if REASONING else ""}_{QUESTION_LEVEL}.jsonl")
print(decompoistion_path)
print(relevant_nodes_path)
input("Confirm?")

#load data
with open(g4_path, "rb") as f:
    nodes = pickle.load(f)


knn = {}
with open(knn_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        knn[line["qid"]] = line["knn"]


question_entities = {}
with open(decompoistion_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        question_entities[line["data_id"]] = line["entities"]

entities_dict = {}
with open(entities_path, "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line)
        entities_dict[line["entity"]] = line["nodes"]

with open(relevant_nodes_path, "w", encoding="utf-8") as f:
    for qid in tqdm(knn.keys()):
        embedding_node_ids = knn[qid][:KNN]
        entity_node_ids = set()
        current_entities = {question_entities[qid]} if isinstance(question_entities[qid], str) else set(question_entities[qid])
        for e in current_entities:
            e = e.upper().strip()
            if e in entities_dict:
                entity_node_ids.update(entities_dict[e])

        for nid in embedding_node_ids: #from V nodes
            if nodes[nid].node_type == "V":
                for edge in nodes[nid].edges:
                    if nodes[edge].node_type == "N":
                        entity_node_ids.add(edge)
        entry_node_ids = set(embedding_node_ids).union(entity_node_ids)

        ppr_search_results = shallow_ppr_local(nodes, entry_node_ids, ppr_context=None, debug=False)
        cross_node_ids = set(ppr_search_results.keys())
        all_nodes_ids = entry_node_ids.union(cross_node_ids)

        reasoning_node_ids = set()
        if REASONING:
            #find shortest paths between entry nodes
            reasoning_entities = [node_id for node_id in entry_node_ids if nodes[node_id].node_type in ['N']]
            paths = all_pairs_shortest_paths(nodes, reasoning_entities, debug = False)
            for index_i in range(len(reasoning_entities)-1):
                for index_j in range(index_i+1, len(reasoning_entities)):
                    i = reasoning_entities[index_i]
                    j = reasoning_entities[index_j]
                    path_ij = paths[i][j]
                    if not path_ij or len(path_ij) <= 2:
                        continue
                    for node_id in path_ij:
                        if node_id not in reasoning_node_ids:
                            reasoning_node_ids.add(node_id)

        data = {
            "qid": qid,
            "KNN": KNN,
            "entry_nodes": list(entry_node_ids),
            "context_nodes": [nid for nid in all_nodes_ids if nodes[nid].node_type not in {"N", "O"}],
            "reasoning_nodes": [nid for nid in reasoning_node_ids if nodes[nid].node_type not in {"N", "O"}],
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
