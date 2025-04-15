import streamlit as st
from openai import OpenAI
from utils import split_text, find_relevant_chunks

st.set_page_config(page_title="Clubspire Chatbot")

# 🔐 Inicializuj OpenAI klienta
client = OpenAI(api_key=st.secrets["openai_api_key"])

# 📘 Načti manuál
@st.cache_data
def load_manual():
    try:
        with open("manual_clubspire.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Chyba při načítání manuálu: {e}")
        return ""

manual_text = load_manual()
if not manual_text:
    st.stop()

# Rozdělení na bloky
chunks = split_text(manual_text)

# 🎯 Titulek
st.title("🤖 Clubspire Chatbot")
st.write("Zeptej se mě na cokoliv ohledně softwaru Clubspire.")

# 🧠 Vstup
user_input = st.text_input("Tvoje otázka:")

if user_input:
    with st.spinner("Přemýšlím..."):
        try:
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

        except Exception as e:
            st.error(f"Chyba při generování odpovědi: {e}")
