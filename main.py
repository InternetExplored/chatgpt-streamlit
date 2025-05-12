import streamlit as st
import openai

# Initialize OpenAI client using secrets
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="ChatGPT with Streamlit", layout="centered")

st.title("💬 Chat with GPT")
st.markdown("Ask me anything!")

# User input
user_input = st.text_input("Your question:")

# Chat completion
if user_input:
    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            message = response.choices[0].message.content
            st.success(message)

        except Exception as e:
            st.error(f"Error: {e}")
