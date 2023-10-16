import streamlit as st
import user_authen.authenticate as authenticate

st.set_page_config(layout="wide")

st.write("Welcome to the dashboard")
authenticate.set_st_state_vars()

if st.session_state["authenticated"]:
    authenticate.button_logout()
else:
    authenticate.button_login()

if (
    st.session_state["authenticated"]
    and "Admin" in st.session_state["user_cognito_groups"]
):
    st.write(
        """You have logged in, you can access other pages"""
    )

else:
    if st.session_state["authenticated"]:
        st.write("You do not have access. Please contact the administrator.")
    else:
        st.write("Please login!")