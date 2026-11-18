
import streamlit as st

from api_client import APIClient, APIClientError


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# API client
# -------------------------------------------------------------------

@st.cache_resource
def get_api_client() -> APIClient:
    return APIClient()


api = get_api_client()


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_conversation_id" not in st.session_state:
    st.session_state.loaded_conversation_id = None


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.title("📚 DocuMind AI")
    st.caption("Your intelligent document assistant")

    st.divider()

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.loaded_conversation_id = None
        st.rerun()

    st.subheader("Recent conversations")

    try:
        result = api.list_conversations(page=1, page_size=50)

        for conversation in result["conversations"]:
            conversation_id = str(conversation["id"])

            if st.button(
                conversation["title"],
                key=f"conversation_{conversation_id}",
                use_container_width=True,
            ):
                st.session_state.conversation_id = conversation_id
                st.session_state.loaded_conversation_id = None
                st.rerun()

    except APIClientError as exc:
        st.error(f"Could not load conversations: {exc}")


# -------------------------------------------------------------------
# Load selected conversation
# -------------------------------------------------------------------

conversation_id = st.session_state.conversation_id

if (
    conversation_id is not None
    and st.session_state.loaded_conversation_id != conversation_id
):
    try:
        conversation = api.get_conversation(conversation_id)

        st.session_state.messages = conversation["messages"]
        st.session_state.loaded_conversation_id = conversation_id

    except APIClientError as exc:
        st.error(f"Could not load conversation: {exc}")


# -------------------------------------------------------------------
# Main chat interface
# -------------------------------------------------------------------

st.title("DocuMind AI")

if not st.session_state.messages:
    st.markdown(
        """
        ### Welcome to DocuMind AI 👋

        Your intelligent document question-answering assistant.

        Ask questions about your uploaded documents and get
        answers grounded in their content.
        """
    )


# Display conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])

        if sources:
            with st.expander("📚 Sources"):
                st.json(sources)


# -------------------------------------------------------------------
# Chat input
# -------------------------------------------------------------------

question = st.chat_input("Ask a question about your documents...")

if question:
    # Display the user's message immediately.
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = api.send_message(
                    question=question,
                    conversation_id=st.session_state.conversation_id,
                )

            answer = response["answer"]
            sources = response.get("sources", [])

            st.markdown(answer)

            if sources:
                with st.expander("📚 Sources"):
                    st.json(sources)

        # Store the generated conversation ID.
        st.session_state.conversation_id = str(
            response["conversation_id"]
        )

        # Store messages locally for immediate rendering.
        st.session_state.messages.extend(
            [
                {
                    "role": "user",
                    "content": question,
                    "sources": [],
                },
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                },
            ]
        )

        st.session_state.loaded_conversation_id = (
            st.session_state.conversation_id
        )

        st.rerun()

    except APIClientError as exc:
        st.error(f"Could not process your question: {exc}")
