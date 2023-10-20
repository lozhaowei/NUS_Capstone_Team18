import streamlit as st
from user_authen.authenticate_components import authenticate, setupemptyprofile, get_role, check_login_status, generate_captcha, \
    generate_session_token, get_role_from_session_token, get_username_from_session_token, login_with_remember_me

st.set_page_config(layout="wide")

st.write("Welcome to the dashboard")

st.header("Login")
if "username" not in st.session_state or "role" not in st.session_state:
    setupemptyprofile()

#maybe do some if statement here so that it is only accessed once????? nt sure
#original_text, captcha_image = generate_captcha()
#st.write(original_text)
#st.image(captcha_image)
#captcha = st.text_input("Enter the letters that you see.").upper()

with st.sidebar:
    st.text("")
    check_login_status()

if st.session_state.username is not None:
    st.write("You have already logged in. Click the side bar to sign out. Thank you!")
else:
    if "session_token" in st.session_state:
        if st.session_state.session_token is not None:
            print("b")
            username = st.text_input("Username", value=get_username_from_session_token(st.session_state.session_token))
            password = st.text_input("Password", type="password", value=st.session_state.password)
            remember_me = st.checkbox(":green[Remember Me]")
            if st.button("Login"):
                st.session_state.username = get_username_from_session_token(st.session_state.session_token)
                st.session_state.role = get_role_from_session_token(st.session_state.session_token)
                st.success("Login successful!")
                st.text("Welcome! You can now navigate through the different pages")
        else:
            login_with_remember_me()
    else:
        login_with_remember_me()


