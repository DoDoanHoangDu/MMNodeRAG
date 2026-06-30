import streamlit as st
import tempfile
import time
import pickle
import faiss
import torch

from LLM.qwen3_vl_embedding import Qwen3VLEmbedder
from LLM.qwen3_vl_reranker import Qwen3VLReranker

from Demo.subquestions_generation import subquestion_generation
from Demo.question_decomposition import question_decomposition
from Demo.knn_retrieval import knn_retrieval
from Demo.graph_retrieval import graph_retrieval
from Demo.rerank_context import rerank_context
from Demo.get_answer import get_answer


#########################################
# Page Config
#########################################

st.set_page_config(
    page_title="GraphRAG Demo",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multimodal GraphRAG Demo")


#########################################
# Load Resources
#########################################

@st.cache_resource
def load_resources():
    # graph
    with open("2-Build_Graph/data/g4.pkl", "rb") as f:
        nodes = pickle.load(f)
    # FAISS
    hnsw = faiss.read_index("2-Build_Graph/data/embeddings_hnsw.faiss")

    with open("2-Build_Graph/data/embedding_processed_ids.txt","r") as f:
        embedding_ids = [line.strip() for line in f]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
    reranker_model = Qwen3VLReranker(model_name_or_path="Qwen/Qwen3-VL-Reranker-2B")

    return (nodes, hnsw, embedding_ids, embedding_model, reranker_model, device)

(nodes, hnsw, embedding_ids, embedding_model, reranker_model, device) = load_resources()
st.success(f"Models loaded on **{device}**")


#########################################
# Sidebar
#########################################

st.sidebar.header("Settings")
knn = st.sidebar.slider("KNN",min_value=1, max_value=20, value=8)
generate_subquestions = st.sidebar.checkbox("Generate Subquestions", value=False)
show_context = st.sidebar.checkbox("Show Retrieved Context", value=True)
show_entities = st.sidebar.checkbox("Show Question Entities", value=True)

#########################################
# Input
#########################################

question = st.text_input("Question", placeholder="Ask anything...")
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

#########################################
# Run Button
#########################################

if st.button("Run"):
    if question == "":
        st.warning("Please enter a question.")
        st.stop()
    image_path = None
    if uploaded_image is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(uploaded_image.read())
        tmp.close()
        image_path = tmp.name
        st.image(image_path, width=350)

    start = time.time()
    total_tokens = 0
    try:
        if generate_subquestions:
            subquestions, token = subquestion_generation(question, image_path)
            if not isinstance(subquestions, list):
                st.error("Subquestion generation failed.")
                st.stop()
            total_tokens += token
            subquestions.append(question)
        else:
            subquestions = [question]

        answers = []
        for i, q in enumerate(subquestions):
            st.divider()
            st.subheader(f"Step {i+1}")
            st.write("### Question")
            st.write(q)
            full_q = "\n".join(answers + [q])
            entities, token = question_decomposition(full_q, image_path)
            if not isinstance(entities, list): 
                st.error("Entity extraction failed")
                st.stop()
            total_tokens += token
            if show_entities:
                st.write("### Question Entities")
                st.write(entities)

            embedding_node_ids = knn_retrieval(embedding_model, hnsw, embedding_ids, knn, full_q, image_path)
            context_nodes = graph_retrieval(nodes, embedding_node_ids, entities)
            reranked_contexts = rerank_context(reranker_model, nodes, full_q, image_path, context_nodes)
            if show_context:
                st.write("### Retrieved Context")
                st.write(reranked_contexts)

            answer, token = get_answer(nodes, full_q, image_path, reranked_contexts)
            if not answer: 
                st.error("Answer failed")
                st.stop()
            total_tokens += token
            answers.append(answer)
            st.write("### Answer")
            st.success(answer)

        st.divider()
        st.header("Final Answer")
        st.success(answers[-1])
        col1, col2 = st.columns(2)
        col1.metric("Total Tokens", total_tokens)
        col2.metric("Time (s)", f"{time.time()-start:.2f}")
    except Exception as e:
        st.exception(e)