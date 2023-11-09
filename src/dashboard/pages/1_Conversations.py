import streamlit as st

from src.dashboard.user_authen.authenticate_components import check_login_status
from src.dashboard.components.page_interface import page

st.set_page_config(layout="wide")

check_login_status()

if st.session_state.role == "admin":
    page('convo', 'Conversations')

else:
    if st.session_state.username is not None:
        st.write("You do not have access. Please contact the administrator.")
    else:
        st.write("Please login!")
