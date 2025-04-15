import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Clubspire Chatbot")

# 🔐 OpenAI API key
client = OpenAI(api_key=st.secrets["openai_api_key"])

# 📘 Načti manuál
@st.cache_data
def load_manual():
    with open("manual_clubspire.txt", "r", encoding="utf-8") as f:
        return f.read()

manual_text = load_manual()

# 🎯 Titulek
st.title("🤖 Clubspire Chatbot")
st.write("Zeptej se mě na cokoliv ohledně softwaru Clubspire.")

# 🧠 Uživatelský vstup
user_input = st.text_input("Tvoje otázka:")

if user_input:
    with st.spinner("Přemýšlím..."):
        prompt = f"""
Jsi technický asistent pro Clubspire. Odpovídáš podle tohoto výňatku z manuálu:

{manual_text[:3000]}

Dotaz: {user_input}
Odpověz česky, výstižně a prakticky.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        st.markdown(f"**Chatbot:** {answer}")
