import streamlit as st
import os
from utils.methods import KatutuboLLMAPI

# Load environment variables for local development
if not bool(os.getenv("PROD")):
    from dotenv import load_dotenv
    load_dotenv(override=True)

# Initialize API
URL = os.getenv("URL")
api = KatutuboLLMAPI(base_url=URL)

st.title("Katutubo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message('assistant'):
    st.markdown("Anong maitutulong ko sa'yo ngayon?")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message('user'):
        st.markdown(message["user"])
    with st.chat_message('assistant'):
        st.markdown(message["assistant"])

# React to user input
if prompt := st.chat_input("What is up?"):
    history = st.session_state.messages
    print(history)

    # Display and store user message
    st.chat_message("user").markdown(prompt)

    # Get response from KatutuboLLMAPI
    api_response = api.infer(prompt, history)

    # Prepare and display bot response
    response = api_response.get("response", "Walang sagot 😅")
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"user": prompt, "assistant": response})
