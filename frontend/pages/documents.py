
import json

import streamlit as st

from api_client import APIClientError
from common import get_api_client


api = get_api_client()


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "document_page" not in st.session_state:
    st.session_state.document_page = 1

if "document_status_filter" not in st.session_state:
    st.session_state.document_status_filter = "All"

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def format_file_size(size: int) -> str:
    """Format file size into a readable representation."""

    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"

    if size >= 1024:
        return f"{size / 1024:.2f} KB"

    return f"{size} bytes"


def format_datetime(value: str | None) -> str:
    """Format an ISO datetime for display."""

    if not value:
        return "—"

    return value.replace("T", " ")[:19]


# -------------------------------------------------------------------
# Page header
# -------------------------------------------------------------------

st.title("📁 Document Library")
st.caption("Upload, browse and monitor your documents.")

st.divider()


# -------------------------------------------------------------------
# Upload documents
# -------------------------------------------------------------------

with st.expander("⬆️ Upload documents", expanded=True):

    with st.form(key=f"upload_form_{st.session_state.upload_key}"):

        uploaded_files = st.file_uploader(
            "Choose documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=f"document_uploader_{st.session_state.upload_key}",
        )

        metadata_input = st.text_area(
            "Document metadata (JSON)",
            value="{}",
            help="Optional metadata shared by all files in this upload.",
        )

        submitted = st.form_submit_button(
            "Upload documents",
            type="primary",
            use_container_width=True,
        )

        if submitted:

            if not uploaded_files:
                st.warning("Please select at least one document.")

            else:
                try:
                    metadata = json.loads(metadata_input)

                    if not isinstance(metadata, dict):
                        st.error("Metadata must be a JSON object.")

                    else:
                        successful_uploads = 0

                        progress = st.progress(0)

                        for index, uploaded_file in enumerate(uploaded_files):

                            try:
                                uploaded_file.seek(0)

                                api.upload_document(
                                    filename=uploaded_file.name,
                                    file_obj=uploaded_file,
                                    metadata=metadata,
                                )

                                successful_uploads += 1

                                st.success(
                                    f"Uploaded: {uploaded_file.name}"
                                )

                            except APIClientError as exc:
                                st.error(
                                    f"Failed to upload "
                                    f"{uploaded_file.name}: {exc}"
                                )

                            progress.progress(
                                (index + 1) / len(uploaded_files)
                            )

                        if successful_uploads:
                            st.session_state.upload_key += 1
                            st.session_state.document_page = 1

                            st.success(
                                f"{successful_uploads} document(s) "
                                "uploaded successfully."
                            )

                except json.JSONDecodeError:
                    st.error("Metadata must contain valid JSON.")


st.divider()


# -------------------------------------------------------------------
# Document library header and filters
# -------------------------------------------------------------------

st.subheader("Your documents")

filter_col, refresh_col = st.columns([3, 1])

with filter_col:
    selected_status = st.selectbox(
        "Filter by processing status",
        options=[
            "All",
            "uploaded",
            "queued",
            "processing",
            "completed",
            "failed",
        ],
        key="document_status_filter",
    )

with refresh_col:
    st.write("")
    refresh = st.button(
        "🔄 Refresh",
        use_container_width=True,
    )

# Reset pagination when the filter changes.
if selected_status != st.session_state.get("last_document_filter"):
    st.session_state.document_page = 1
    st.session_state.last_document_filter = selected_status


# -------------------------------------------------------------------
# Retrieve documents
# -------------------------------------------------------------------

status_filter = (
    None if selected_status == "All" else selected_status
)

try:
    result = api.list_documents(
        page=st.session_state.document_page,
        page_size=20,
        status_filter=status_filter,
    )

    documents = result["documents"]
    total = result["total"]
    page = result["page"]
    page_size = result["page_size"]

    total_pages = max(1, (total + page_size - 1) // page_size)

    # Summary metrics.
    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Total documents", total)

    with metric_col2:
        st.metric("Current page", f"{page} / {total_pages}")

    st.divider()

    # Display document list.
    if not documents:
        st.info(
            "No documents found. Upload a document to get started."
        )

    else:
        for document in documents:

            with st.container(border=True):

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**📄 {document['filename']}**")

                    st.caption(
                        f"ID: {document['id']}"
                    )

                    st.caption(
                        f"{format_file_size(document['file_size_bytes'])}"
                        f" · Uploaded {format_datetime(document['created_at'])}"
                    )

                with col2:
                    document_status = document["status"]

                    status_colors = {
                        "uploaded": "gray",
                        "queued": "orange",
                        "processing": "orange",
                        "completed": "green",
                        "failed": "red",
                    }

                    st.markdown(
                        f":{status_colors.get(document_status, 'gray')}"
                        f"[{document_status.upper()}]"
                    )

                details_col1, details_col2, details_col3 = st.columns(3)

                with details_col1:
                    st.caption(
                        f"Pages: {document.get('page_count') or '—'}"
                    )

                with details_col2:
                    st.caption(
                        f"Chunks: {document.get('chunk_count') or '—'}"
                    )

                with details_col3:
                    st.caption(
                        f"Updated: "
                        f"{format_datetime(document.get('updated_at'))}"
                    )

                if document.get("error_message"):
                    with st.expander("Processing error"):
                        st.error(document["error_message"])

                with st.expander("Document metadata"):
                    st.json(document.get("metadata") or {})


    # ----------------------------------------------------------------
    # Pagination
    # ----------------------------------------------------------------

    if total_pages > 1:

        previous_col, page_col, next_col = st.columns(
            [1, 2, 1]
        )

        with previous_col:
            if st.button(
                "← Previous",
                disabled=page <= 1,
                use_container_width=True,
            ):
                st.session_state.document_page -= 1
                st.rerun()

        with page_col:
            st.markdown(
                f"<div style='text-align:center'>"
                f"Page {page} of {total_pages}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with next_col:
            if st.button(
                "Next →",
                disabled=page >= total_pages,
                use_container_width=True,
            ):
                st.session_state.document_page += 1
                st.rerun()


except APIClientError as exc:
    st.error(f"Could not retrieve documents: {exc}")
