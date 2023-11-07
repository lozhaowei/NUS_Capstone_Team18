import streamlit as st

from src.dashboard.data.spark_pipeline import SparkPipeline

st.set_page_config(layout="wide")

from user_authen.authenticate_components import setupemptyprofile, get_role, check_login_status, \
    generate_captcha, get_manager, \
    login_with_remember_me, cookie_manager

st.write("Welcome to the dashboard")

st.header(":violet[Login Page]")
if "username" not in st.session_state or "role" not in st.session_state:
    setupemptyprofile()

# maybe do some if statement here so that it is only accessed once????? nt sure
# original_text, captcha_image = generate_captcha()
# st.write(original_text)
# st.image(captcha_image)
# captcha = st.text_input("Enter the letters that you see.").upper()

with st.sidebar:
    st.text("")
    check_login_status()

#st.write(cookie_manager.get_all())
cookie_manager.get_all()

if st.session_state.username is not None:
    st.write("You have already logged in. Click the side bar to sign out. Thank you!")
else:
    #if remembered
    username = cookie_manager.get(cookie="username")
    #print(username)
    if cookie_manager.get(cookie="username"):
        st.write(f"Do you want to login to the account: {username}?")
        if st.button(":green[Login]"):
            st.session_state.username = username
            st.session_state.role = get_role(username)
            st.success("Login successful!")
            st.text("Welcome! You can now navigate through the different pages")

            # Run query recommended item hit ratio functions
            spark_pipeline = SparkPipeline()
            spark_pipeline.initialize_spark_session()
            spark_pipeline.run_video_upvote_percentage_pipeline()
            spark_pipeline.run_conversation_like_percentage_pipeline()
            spark_pipeline.close_spark_session()
        if st.button(":orange[Login to another account]"):
            st.warning("Your previously remembered account will be removed from cache")
            cookie_manager.delete("username")
            login_with_remember_me()
    #if not remembered, just go through the normal login process
    else:
        login_with_remember_me()
