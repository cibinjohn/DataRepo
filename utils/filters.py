import streamlit as st
import pandas as pd

# ---------------------------------------------
# Predefined time range mapping (label → days)
# Used to convert UI selection into a numeric
# window for filtering data
# ---------------------------------------------
TIME_RANGES = {
    "Last day": 1,
    "Last week": 7,
    "Last month": 30,
    "Last 3 months": 90,
    "Last year": 365
}


# ---------------------------------------------
# Initialize session state filters
# Ensures Streamlit does not throw key errors
# and provides default filter values on first load
# ---------------------------------------------
def init_filters():

    # Default time range selection
    if "time_range" not in st.session_state:
        st.session_state.time_range = "Last week"

    # Default DAG selection (used for filtering views)
    if "selected_dag" not in st.session_state:
        st.session_state.selected_dag = "All"


# ---------------------------------------------
# Apply time-based filter on a DataFrame
# Assumes the date column is either datetime
# or convertible to datetime
#
# Uses Streamlit session state to determine
# selected time range
# ---------------------------------------------
def apply_time_filter(df, date_col="run_date"):

    # Work on a copy to avoid mutating original data
    df = df.copy()

    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Get number of days from selected UI filter
    days = TIME_RANGES[st.session_state.time_range]

    # Compute cutoff timestamp
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)

    # Return only rows within the selected time window
    return df[df[date_col] >= cutoff]