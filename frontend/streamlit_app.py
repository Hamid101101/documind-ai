
import streamlit as st


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

chat_page = st.Page(
    "pages/chat.py",
    title="Chat",
    icon="💬",
    default=True,
)

documents_page = st.Page(
    "pages/documents.py",
    title="Documents",
    icon="📁",
)

navigation = st.navigation(
    [chat_page, documents_page],
    position="sidebar",
)

navigation.run()
