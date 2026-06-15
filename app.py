import streamlit as st
from utils.data_loader import load_data
import pandas as pd

st.set_page_config(
    page_title="DataOps Assistant",
    page_icon="🏠",
    layout="wide",
)

def render_global_filters():
    data = load_data()
    dag_options = ["All"] + sorted(data["dag_runs"]["dag_id"].unique())
    today = pd.Timestamp.today().normalize().date()

    with st.sidebar:
        st.header("Filters")

        default_from, default_to = st.session_state.get(
            "date_range", (today - pd.Timedelta(days=7), today)
        )
        c1, c2 = st.columns(2)
        from_date = c1.date_input("From", value=default_from)
        to_date = c2.date_input("To", value=default_to)
        if from_date > to_date:
            from_date, to_date = to_date, from_date
        st.session_state["date_range"] = (from_date, to_date)

        current = st.session_state.get("selected_dag", "All")
        st.session_state["selected_dag"] = st.selectbox(
            "DAG", dag_options,
            index=dag_options.index(current) if current in dag_options else 0,
        )

render_global_filters()

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
    st.Page("pages/4_RCA_Report.py", title="RCA Reports", icon="📄"),
    st.Page(historical_explorer,                  title="Historical Explorer", icon="📊"),
]

pg = st.navigation(pages, position="top")
pg.run()