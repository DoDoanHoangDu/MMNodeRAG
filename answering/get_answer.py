import time
start = time.time()
import os
from sentence_transformers import SentenceTransformer
import torch
import pickle
import json
import faiss
from google import genai
import pandas as pd
import numpy as np
from answering.get_context import get_context
from answering.answer_prompt import answer_prompt


#set file paths
dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(dir_path)

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("google/embeddinggemma-300m", device=device)

#Load Data
#Graph - Node dict
with open(f"{root_path}/graphs/data/graphs/G6_entity_and_chunk_resolution_graph.pkl", "rb") as f:
    medical_graph = pickle.load(f)

#Embeddings
hnsw = faiss.read_index(f"{root_path}/graphs/data/embedding/medical_index.faiss")
with open(f"{root_path}/graphs/data/embedding/medical_ids.json", "r") as f:
    medical_embedding_ids = json.load(f)
num_vectors = hnsw.ntotal
dimension = hnsw.d
embeddings = np.zeros((num_vectors, dimension), dtype='float32')
for i in range(num_vectors):
    embeddings[i] = hnsw.reconstruct(i)

#Entities
with open(f"{root_path}/graphs/data/nodes/entity/medical_entities.pkl", "rb") as f:
    medical_entities = pickle.load(f)
medical_overview = pd.read_parquet(f"{root_path}/graphs/data/nodes/community/medical_communities_overview.parquet")

#LLM api call
with open(f"{root_path}/API_KEY.txt", "r", encoding="utf-8") as f:
    API_KEY = f.read()

def call_gemini(text):
    client = genai.Client(api_key = API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text
    )
    return response

#Context setup
graph_context = {
    'graph': medical_graph,
    'entities': medical_entities,
    'overview': medical_overview
}

embedding_context = {
    'model': model,
    'index': hnsw,
    'ids': medical_embedding_ids,
}

print(f"Load time: {time.time() - start:.2f} seconds.")
#questioning loop
loop_sep = "#"*100 + "\n"
while True:
    try:
        reasoning = False
        question = input("Enter your medical question (or 'quit' to quit): ")
        if question.lower() == 'quit':
            print(loop_sep)
            break
        
        reason_input = input("Enable reasoning mode? (y/n): ").strip().lower()
        if reason_input == 'y':
            reasoning = True
        elif reason_input == 'n':
            reasoning = False
        else:
            print("Invalid input for reasoning mode. Defaulting to 'n'.")
            reasoning = False
        print("-"*100)
        start = time.time()

        ppr_context = {
            'k_ppr': None,
            'alpha': 0.5,
            't':5
        }
        query_context = {
            'question': question,
            'k_embedding': 8,
            'ppr': ppr_context
        }

        context = get_context(query_context, graph_context, embedding_context, API_KEY, True, reasoning)
        with open(f"{root_path}/answering/context.txt","w",  encoding="utf-8") as f:
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
            full_context = "\n\n".join(context.values())
            prompt = answer_prompt(full_context, question)
            response = call_gemini(prompt)
            answer = response.text
            print("Answer Generated")
            #print(answer)
            with open(f"{root_path}/answering/answer.txt","w") as f:
                f.write(answer)
            #print("-"*100)
            usage = response.usage_metadata
            print(f"Prompt Tokens (Input): {usage.prompt_token_count}")
            print(f"Candidate Tokens (Output): {usage.candidates_token_count}")
            print(f"Total Tokens: {usage.total_token_count}")
        print(f"Answer generation time: {time.time() - finish_retrieval_time:.2f} seconds.")
        print(f"Total time taken: {time.time() - start:.2f} seconds.")
    except Exception as e:
        print("An error occurred:", str(e))
    print(loop_sep*10)