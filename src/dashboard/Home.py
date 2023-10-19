import streamlit as st
from user_authen.authenticate_components import authenticate, setupemptyprofile, get_role, check_login_status, generate_captcha

st.set_page_config(layout="wide")

st.write("Welcome to the dashboard")

st.header("Login")
if "username" not in st.session_state or "role" not in st.session_state:
    setupemptyprofile()

username = st.text_input("Username")
password = st.text_input("Password", type="password")

#maybe do some if statement here so that it is only accessed once????? nt sure
#original_text, captcha_image = generate_captcha()
#st.write(original_text)
#st.image(captcha_image)
#captcha = st.text_input("Enter the letters that you see.").upper()

with st.sidebar:
    st.text("")
    check_login_status()

if st.button("Login"):
    #print(captcha, original_text, captcha_image)
    #if captcha == original_text:
    if authenticate(username, password):
        st.session_state.username = username
        st.session_state.role = get_role(username)
        st.success("Login successful!")
        st.text("Welcome! You can now navigate through the different pages")
    else:
        st.error("Login failed. Please check your credentials and/or your CAPTCHA.")
