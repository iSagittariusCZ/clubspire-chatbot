import streamlit as st
from openai import OpenAI
from utils import split_text, find_relevant_chunks

st.set_page_config(page_title="Clubspire Chatbot")

# Inicializuj klienta s API klíčem
client = OpenAI(api_key=st.secrets["openai_api_key"])

# 📘 Načti manuál
@st.cache_data
def load_manual():
    with open("manual_clubspire.txt", "r", encoding="utf-8") as f:
        return f.read()

manual_text = load_manual()
chunks = split_text(manual_text)

# 🎯 Titulek
st.title("🤖 Clubspire Chatbot")
st.write("Zeptej se mě na cokoliv ohledně softwaru Clubspire.")

# 🧠 Uživatelský vstup
user_input = st.text_input("Tvoje otázka:")

if user_input:
    with st.spinner("Přemýšlím..."):
        relevant_chunks = find_relevant_chunks(client, chunks, user_input, top_n=1)
        context = "\n\n".join(relevant_chunks)

        prompt = f"""
Jsi technický asistent pro software Clubspire. Máš k dispozici následující výňatek z manuálu:

\"\"\"{context}\"\"\"

Na základě uvedeného textu odpověz výhradně podle něj. Pokud odpověď v textu není, napiš: 'V manuálu se tato informace nenachází.'

Dotaz: {user_input}
Odpověz česky, prakticky a přesně.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        st.markdown(f"**Chatbot:** {answer}")
