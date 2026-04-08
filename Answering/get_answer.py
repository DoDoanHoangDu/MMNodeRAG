import time
start = time.time()
import os
from sentence_transformers import SentenceTransformer
import torch
import pickle
import json
import faiss
import numpy as np
from Answering.get_context import get_context
from LLM.prompts.answer_prompt import answer_prompt
from LLM.call_api import call_api

def format_list(l):
    ans = []
    for i in range(len(l)):
        ans.append(f"[{i+1}] {l[i]}")
    return "\n".join(ans)

#set file paths
dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(dir_path)

# Load embedding model


#Load Data
#Graph - Node dict
with open(f"{root_path}/2-Build_Graph/data/g4.pkl", "rb") as f:
    graph = pickle.load(f)

#Embeddings
hnsw = faiss.read_index(f"{root_path}/2-Build_Graph/data/embedding.faiss")
with open(f"{root_path}/2-Build_Graph/data/embedding_ids.json", "r") as f:
    embedding_ids = json.load(f)
num_vectors = hnsw.ntotal
dimension = hnsw.d
embeddings = np.zeros((num_vectors, dimension), dtype='float32')
for i in range(num_vectors):
    embeddings[i] = hnsw.reconstruct(i)

#Entities
with open(f"{root_path}/2-Build_Graph/data/entities.json", "rb") as f:
    entities = json.load(f)

#Context setup
graph_context = {
    'graph': graph,
    'entities': entities
}

embedding_context = {
    'model': model,
    'index': hnsw,
    'ids': embedding_ids,
}

print(f"Load time: {time.time() - start:.2f} seconds.")
#questioning loop
loop_sep = "#"*100 + "\n"
while True:
    try:
        reasoning = False
        question = input("Enter your question (or 'quit' to quit): ")
        if question.lower() == 'quit':
            print(loop_sep)
            break
        
        reason_input = input("Enable reasoning mode? (y/n): ").strip().upper()
        if reason_input == 'Y':
            reasoning = True
        elif reason_input == 'N':
            reasoning = False
        else:
            print("Invalid input for reasoning mode. Defaulting to 'n'.")
            reasoning = False
        print("-"*100)
        start = time.time()

        ppr_context = {
            'k_ppr': None,
            'alpha': 0.5,
            't':3
        }
        query_context = {
            'question': question,
            'k_embedding': 8,
            'ppr': ppr_context
        }

        context = get_context(query_context, graph_context, embedding_context, debug=True, reasoning=reasoning)
        with open(f"{root_path}/Answering/context.txt","w",  encoding="utf-8") as f:
            c = 1
            for key in context:
                f.write(f"Context {c}/{len(context.keys())}: \n")
                f.write("Node ID: {}\n".format(key))
                f.write("Content: {}\n".format(context[key]))
                f.write("-" * 100 + "\n")
                c += 1
        finish_retrieval_time = time.time()
        print(f"Total retrieval time: {finish_retrieval_time - start:.2f} seconds.")
        print("Number of retrieved context nodes:", len(context))
        print("-"*100)
        #Generate answer
        generate_answer = False
        if generate_answer:
            full_context = format_list(context)
            prompt = answer_prompt(full_context, question)
            response, token = call_api(prompt, model="gemini--flash", mode="gemini")
            print("Answer Generated")
            #print(answer)
            with open(f"{root_path}/Answering/answer.txt","w") as f:
                f.write(response)
            #print("-"*100)
            print(f"Total Tokens: {token}")
        print(f"Answer generation time: {time.time() - finish_retrieval_time:.2f} seconds.")
        print(f"Total time taken: {time.time() - start:.2f} seconds.")
    except Exception as e:
        print("An error occurred:", str(e))
    print(loop_sep*10)