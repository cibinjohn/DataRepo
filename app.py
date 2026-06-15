import streamlit as st

st.set_page_config(
    page_title="DataOps Assistant",
    page_icon="🏠",
    layout="wide",
)

st.markdown("""
<style>
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] a span,
header[data-testid="stHeader"] nav a,
header[data-testid="stHeader"] nav span {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}
/* give the taller text room so it isn't clipped */
header[data-testid="stHeader"] {
    height: 4rem;
}
</style>
""", unsafe_allow_html=True)

# Inline pages for the ones without their own file yet
def rca_reports():
    st.title("RCA Reports")

def historical_explorer():
    st.title("Historical Explorer")

pages = [
    st.Page("pages/1_Home.py",                   title="Home",                icon="🏠", default=True),
    st.Page("pages/2_Failure_Inbox.py",          title="Failure Inbox",       icon="📥"),
    st.Page("pages/3_Investigation_Workspace.py", title="Investigation",      icon="🔍"),
    st.Page(rca_reports,                          title="RCA Reports",         icon="📄"),
    st.Page(historical_explorer,                  title="Historical Explorer", icon="📊"),
]

pg = st.navigation(pages, position="top")
pg.run()