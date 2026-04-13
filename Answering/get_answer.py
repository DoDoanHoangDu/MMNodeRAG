import time
start = time.time()
import os
import torch
import pickle
import json
import faiss
import numpy as np
from Answering.get_context import get_context
from LLM.prompts.answer_prompt import answer_prompt
from LLM.call_api import call_api
from LLM.qwen3_vl_embedding import Qwen3VLEmbedder

def format_list(l):
    ans = []
    for i in range(len(l)):
        ans.append(f"[{i+1}] {l[i]}")
    return "\n".join(ans)

def is_image_file(path):
    if not os.path.isfile(path):
        return False
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
    return path.lower().endswith(valid_extensions)

#set file paths
dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(dir_path)

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

#Load Data
#Graph - Node dict
with open(f"{root_path}/2-Build_Graph/data/g4.pkl", "rb") as f:
    graph = pickle.load(f)

#Embeddings
hnsw = faiss.read_index(f"{root_path}/2-Build_Graph/data/embeddings_hnsw.faiss")
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
        if question.lower().strip() == 'quit':
            print(loop_sep)
            break
        image_input = input("Enter your image path: ")
        if not image_input.strip():
            image_input = None
        elif not os.path.exists(image_input):
            raise ValueError("Image path does not exist")
        elif not is_image_file(image_input):
            raise ValueError("Invalid file path")
        
        reasoning_input = input("Enable reasoning mode? (y/n): ").strip().upper()
        if reasoning_input == 'Y':
            reasoning = True
        elif reasoning_input == 'N':
            reasoning = False
        else:
            print("Invalid input for reasoning mode. Defaulting to 'n'.")
            reasoning = False
        full_question = [{"text": question, "image": image_input, "instruction": "Retrieve images or text relevant to the user's query."}]
        print("-"*100)
        start = time.time()

        ppr_context = {
            'k_ppr': None,
            'alpha': 0.5,
            't':3
        }
        query_context = {
            'question': full_question,
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
            prompt = answer_prompt(full_context, full_question)
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