
import streamlit as st

from api_client import APIClient


@st.cache_resource
def get_api_client() -> APIClient:
    return APIClient()
