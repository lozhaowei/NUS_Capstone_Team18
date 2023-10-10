import streamlit as st
import numpy as np
import user_authen.authenticate as authenticate

# Page configuration
st.set_page_config(page_title="1.Test")
# Check authentication
authenticate.set_st_state_vars()
# Add login/logout buttons
#print(st.session_state, "abc")

if st.session_state["authenticated"]:
    authenticate.button_logout()
else:
    authenticate.button_login()
# Rest of the page
st.markdown("# Test Login Page")
st.sidebar.header("Test Login Page")

if (
    st.session_state["authenticated"]
    and "Admin" in st.session_state["user_cognito_groups"]
):
    st.write(
        """This demo illustrates a combination of plotting and animation with
    Streamlit. We're generating a bunch of random numbers in a loop for around
    5 seconds. Enjoy!"""
    )

else:
    if st.session_state["authenticated"]:
        st.write("You do not have access. Please contact the administrator.")
    else:
        st.write("Please login!")