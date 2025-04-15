import streamlit as st
from openai import OpenAI
from utils import split_text, find_relevant_chunks

st.set_page_config(page_title="Clubspire Chatbot")
client = OpenAI(api_key=st.secrets["openai_api_key"])

# 🗂️ Načti manuál
@st.cache_data
def load_manual():
    with open("manual_clubspire.txt", "r", encoding="utf-8") as f:
        return f.read()

manual_text = load_manual()

# Rozdělení na kratší části
chunks = split_text(manual_text, chunk_size=1000, overlap=200)

# 🧠 Vstup uživatele
st.title("🤖 Clubspire Chatbot")
user_input = st.text_input("Tvoje otázka:")

if user_input:
    with st.spinner("Přemýšlím..."):
        try:
            relevant_chunks = find_relevant_chunks(client, chunks, user_input, top_n=2, limit_chunks=5)
            context = "\n\n".join(relevant_chunks)

            prompt = f"""
Jsi technický asistent pro software Clubspire. Níže máš výňatek z manuálu:

\"\"\"{context}\"\"\"

Na základě uvedeného textu odpověz na dotaz níže co nejpřesněji a prakticky. Pokud odpověď v textu není, napiš: 'V manuálu se tato informace nenachází.'

Dotaz: {user_input}
Odpověz česky a konkrétně:
"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )

            answer = response.choices[0].message.content
            st.markdown(f"**Chatbot:** {answer}")

        except Exception as e:
            st.error(f"❌ Chyba: {e}")
