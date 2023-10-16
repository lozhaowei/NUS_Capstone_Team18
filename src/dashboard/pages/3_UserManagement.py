import streamlit as st
import numpy as np
import user_authen.authenticate as authenticate
import os
import boto3

# Page configuration
st.set_page_config(page_title="3. User Management")
# Check authentication
authenticate.set_st_state_vars()
# Add login/logout buttons
#print(st.session_state, "abc")

st.markdown("# User Management")

if (
    st.session_state["authenticated"]
    and "Admin" in st.session_state["user_cognito_groups"]
):
    st.write(
        """This section will be filled with the user details for admin users to manage"""
    )

else:
    if st.session_state["authenticated"]:
        st.write("You do not have access. Please contact the administrator.")
    else:
        st.write("Please login!")


