import streamlit as st

st.set_page_config(layout="wide")

from src.dashboard.data.spark_pipeline import SparkPipeline
from user_authen.authenticate_components import set_up_empty_profile, get_role, check_login_status, \
    login_with_remember_me, cookie_manager

st.write("Welcome to the dashboard")

# Instantiate an empty session state
st.header(":violet[Login Page]")
if "username" not in st.session_state or "role" not in st.session_state:
    set_up_empty_profile()

# Check the login status to display the "Log In"/"Log Out" buttons
with st.sidebar:
    st.text("")
    check_login_status()

# Obtain the data from cookie
cookie_manager.get_all()

# Check if the user has logged in, if yes, then display that he/she has logged in
# If not, check the cookie to see if there was user account remembered
# If yes, then ask if the user wants to login through that account
# If there is no account remembered or if the user wants to login using another account, he will de directed to the login page
if st.session_state.username is not None:
    st.write("You have already logged in. Click the side bar to sign out. Thank you!")
else:
    # if remembered
    username = cookie_manager.get(cookie="username")
    # print(username)
    if cookie_manager.get(cookie="username"):
        st.write(f"Do you want to login to the account: {username}?")

        if st.button(":green[Login]"):
            st.session_state.username = username
            st.session_state.role = get_role(username)
            st.success("Login successful!")
            st.text("Welcome! You can now navigate through the different pages")

        if st.button(":orange[Login to another account]"):
            st.warning("Your previously remembered account will be removed from cache")
            cookie_manager.delete("username")
            login_with_remember_me()
    # if not remembered, just go through the normal login process
    else:
        login_with_remember_me()
