from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def find_relevant_chunks(client, chunks, question, top_n=1, limit_chunks=5):
    st.write(f"🔍 Hledám v {min(len(chunks), limit_chunks)} z {len(chunks)} chunků")

    chunk_embeddings = []

    # Embeddingy pro omezený počet chunků (např. 5)
    for i, chunk in enumerate(chunks[:limit_chunks]):
        st.write(f"📦 Vytvářím embedding pro chunk {i + 1}")
        try:
            emb = client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            ).data[0].embedding
            chunk_embeddings.append(emb)
        except Exception as e:
            st.error(f"❌ Chyba při embeddingu chunku {i + 1}: {e}")
            chunk_embeddings.append([0.0] * 1536)  # fallback embedding

    # Embedding pro dotaz
    st.write("❓ Vytvářím embedding dotazu...")
    try:
        question_embedding = client.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        ).data[0].embedding
    except Exception as e:
        st.error(f"❌ Chyba při embeddingu dotazu: {e}")
        return []

    st.write("📏 Počítám podobnosti...")
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    return [chunks[i] for i in top_indices]
