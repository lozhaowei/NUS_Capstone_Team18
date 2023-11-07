import streamlit as st
import numpy as np
from user_authen.authenticate_components import user_data, user_update, sign_up, check_login_status, change_password


st.set_page_config(layout="wide")
st.header(":violet[User Management]")
st.markdown("""---""")

check_login_status()
#print('username' in st.session_state)
if st.session_state.role == "admin":
    st.subheader(":orange[User Data]")
    if st.checkbox(":green[Show User Data]"):
        st.write(user_data)

    st.markdown("""---""")
    st.subheader(":orange[Add User]")
    sign_up()

    st.markdown("""---""")
    st.subheader(":orange[Manage Users]")
    user_update()

elif 'username' in st.session_state:
    if st.session_state.username:
        st.write("You can change your password here")
        change_password()
    else:
        st.write("Please login!!")

else:
    st.write("Please login!!")