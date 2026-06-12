import streamlit as st
import pandas as pd


# ---------------------------------------------
# Initialize session state filters
# ---------------------------------------------
def init_filters():
    if "date_range" not in st.session_state:
        today = pd.Timestamp.today().normalize().date()
        st.session_state.date_range = (today - pd.Timedelta(days=7), today)

    if "selected_dag" not in st.session_state:
        st.session_state.selected_dag = "All"


# ---------------------------------------------
# Apply from/to date filter on a DataFrame.
# Reads st.session_state.date_range = (from_date, to_date).
# The 'to' day is included in full (up to 23:59:59).
# ---------------------------------------------
def apply_date_filter(df, date_col="run_date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    from_date, to_date = st.session_state.date_range
    from_ts = pd.Timestamp(from_date)
    to_ts = pd.Timestamp(to_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    return df[(df[date_col] >= from_ts) & (df[date_col] <= to_ts)]