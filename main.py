import streamlit as st
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chat with GPT", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Chat with GPT")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
