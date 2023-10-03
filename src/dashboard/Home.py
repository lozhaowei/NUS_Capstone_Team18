import streamlit as st
from src.dashboard.data.data_handling import load_data

st.set_page_config(layout="wide")

st.write("Welcome to the dashboard")

if (
    st.session_state["authenticated"]
    and "Admin" in st.session_state["user_cognito_groups"]
):
    metrics_data = load_data().reset_index(drop=True)
    st.write(metrics_data)

else:
    if st.session_state["authenticated"]:
        st.write("You do not have access. Please contact the administrator. But here is a sample for what you can see as GUEST :)")
    else:
        st.write("Please login!")

