import streamlit as st
from chatbot import chatbot

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 My RAG Chatbot")

st.write("Ask questions about your PDF documents.")

question = st.chat_input("Ask something...")

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chatbot(question)

        st.write(answer)