import streamlit as st
import numpy as np
from user_authen.authenticate_components import user_data, user_update, sign_up, check_login_status


st.set_page_config(layout="wide")
st.header("User Management")
st.markdown("""---""")

check_login_status()

if st.session_state.role == "admin":
    if st.checkbox("Show User Data"):
        st.write(user_data)

    st.markdown("""---""")
    st.subheader("Add User")
    sign_up()

    st.markdown("""---""")
    user_update()

elif st.session_state.username:
    st.write("You do not have access to this page")
else:
    st.write("Please login!!")