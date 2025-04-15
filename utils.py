from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def find_relevant_chunks(client: OpenAI, chunks, question, top_n=1):
    # Vygeneruj embeddingy pro všechny chunk části
    chunk_embeddings = [
        client.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding
        for chunk in chunks
    ]
    
    # Vygeneruj embedding pro dotaz
    question_embedding = client.embeddings.create(input=question, model="text-embedding-ada-002").data[0].embedding
    
    # Vypočítej kosinovou podobnost
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    return [chunks[i] for i in top_indices]
