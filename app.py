import streamlit as st
from openai import OpenAI
from utils import split_text, find_relevant_chunks

st.set_page_config(page_title="Clubspire Chatbot")

# 🔐 OpenAI API key
client = OpenAI(api_key=st.secrets["openai_api_key"])

# 📘 Načti manuál
@st.cache_data
def load_manual():
    with open("manual_clubspire.txt", "r", encoding="utf-8") as f:
        return f.read()

manual_text = load_manual()

# Rozděl text na části
chunks = split_text(manual_text)

# 🎯 Titulek
st.title("🤖 Clubspire Chatbot")
st.write("Zeptej se mě na cokoliv ohledně softwaru Clubspire.")

# 🧠 Uživatelský vstup
user_input = st.text_input("Tvoje otázka:")

if user_input:
    with st.spinner("Přemýšlím..."):
        try:
            # Vyhledej nejrelevantnější úryvky z manuálu
            relevant_chunks = find_relevant_chunks(client, chunks, user_input, top_n=2)
            st.subheader("🔍 Nejrelevantnější výňatky z manuálu:")
            for i, chunk in enumerate(relevant_chunks):
                st.code(chunk, language="markdown")

            prompt = f"""
Jsi technický asistent pro software Clubspire. Máš k dispozici následující výňatky z manuálu:

\"\"\"{relevant_chunks[0]}\n\n{relevant_chunks[1]}\"\"\"

Na základě uvedeného textu odpověz výhradně podle něj. Pokud odpověď v textu není, napiš: 'V manuálu se tato informace nenachází.'

Dotaz: {user_input}
Odpověz česky, prakticky a přesně.
"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )

            answer = response.choices[0].message.content
            st.subheader("💬 Chatbot:")
            st.markdown(answer)

        except Exception as e:
            st.error(f"Nastala chyba: {e}")
