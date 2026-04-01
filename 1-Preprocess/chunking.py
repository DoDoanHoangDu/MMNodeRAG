import json
import spacy
import os
#Chunking logic
print("Starting chunking process...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
nlp.enable_pipe("senter")
 
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]
 
def chunking(text, min_chunk_size = 150, max_chunk_size = 300, overlap = 50):
    chunks = []
    #Extract paragraphs list p from text
    paragraphs = text.split("\n")
    m = len(paragraphs)
    p = []
    residual = ""
    for i in range(m):
        paragraph = paragraphs[i].strip()
        if residual:
            paragraph = residual + "\n" + paragraph
        words = paragraph.split()
        n = len(words)
        if n == 0:
            continue
        elif n < min_chunk_size: #short paragraph, could be the conclusion of previous, or introduction of next
            residual = paragraph
            if len(p) > 0:
                p[-1] = p[-1] + "\n" + paragraph
            elif i == m - 1:
                p.append(paragraph)
        else:
            residual = ""
            p.append(paragraph)
 
    #Extract chunks from each paragraph from p
    for paragraph in p:
        if len(paragraph.split()) <= max_chunk_size:
            chunks.append(paragraph)
            continue
        sents = split_sentences(paragraph)
        current_chunk = []
        buffer = []
        current_count = 0
        buffer_count = 0
        for sent in sents:
            sent_count = len(sent.split())
            if current_count + sent_count < max_chunk_size:
                current_chunk.append(sent)
                current_count += sent_count
                buffer.append(sent)
                buffer_count += sent_count
                while buffer_count - len(buffer[0].split()) >= overlap:
                    buffer_count -= len(buffer[0].split())
                    buffer = buffer[1:]
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = buffer.copy()
                current_count = buffer_count
        if current_chunk and current_count > buffer_count:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
    return chunks
 
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)
 
#Check for processed document ids
ids_path = os.path.join(DIR_PATH, "data/processed_ids.txt")
processed_ids = set()
try:
    with open(ids_path, "r") as f:
        for line in f:
            processed_ids.add(line.strip())
except FileNotFoundError:
	pass
 
#Read data and run all logic line by line
print("Reading corpus and processing...")
chunks_path = os.path.join(DIR_PATH, "data/chunks.jsonl")
corpus_path = os.path.normpath(os.path.join(BASE_DIR, "InfoSeek", "KnowledgeBase.jsonl"))
with open(corpus_path, "r", encoding = "utf-8") as fin, \
    open(chunks_path, "a", encoding="utf-8") as fout, \
    open(ids_path, "a") as idfile:
     
    def write(chunk_id, chunk):
        record = {
            "chunk_id": chunk_id,
            "chunk_content": chunk
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    progress = 0
    for line in fin:
        progress += 1
        record = json.loads(line)
        rid = str(record["wikidata_id"])

        if rid in processed_ids:
            continue

        content = record.get("wikipedia_content", "")
        if not content:
            continue
        chunks = chunking(content)
        for index, chunk in enumerate(chunks):
            chunk_id = f"{rid}:T{index:03d}"
            write(chunk_id, chunk.strip())

        idfile.write(rid + "\n")
        idfile.flush()
        processed_ids.add(rid)
        if progress % 10000 == 0:
            print(f"Processed {progress} documents.")
 