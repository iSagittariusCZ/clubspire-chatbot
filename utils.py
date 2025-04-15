import numpy as np
import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

# Funkce pro rozřezání textu na bloky (např. 500 tokenů)
def split_text(text, chunk_size=500, overlap=100):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        decoded = encoding.decode(chunk)
        chunks.append(decoded)

    return chunks

# Funkce pro výpočet podobnosti mezi dotazem a bloky
def find_relevant_chunks(chunks, question, top_n=1):
    embeddings = [client.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding for chunk in chunks]
    question_emb = client.embeddings.create(input=question, model="text-embedding-ada-002").data[0].embedding

    sims = cosine_similarity([question_emb], embeddings)[0]
    top_indices = sims.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]
