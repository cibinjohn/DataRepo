import streamlit as st
from utils.filters import init_filters, TIME_RANGES

st.set_page_config(
    page_title="DataOps Assistant",
    page_icon="🔎",
    layout="wide"
)

init_filters()


st.sidebar.divider()

st.title("🔎 DataOps Assistant")

st.markdown("""
AI-powered DAG Failure Investigation Workspace

Use the sidebar to navigate:

- Workspace Home
- Failure Inbox
- Investigation Workspace
- RCA Report
- Historical Explorer
""")