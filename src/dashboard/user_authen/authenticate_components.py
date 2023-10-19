import pandas as pd
import streamlit as st
import hashlib
import os
import re
from streamlit_extras.switch_page_button import switch_page
from captcha.image import ImageCaptcha
from PIL import Image
import io
import base64
import random

# Define the path for the CSV file
csv_file_path = "user_data.csv"

# Create an empty DataFrame to store user details or load from CSV if it exists
if os.path.exists(csv_file_path):
    user_data = pd.read_csv(csv_file_path)
else:
    user_data = pd.DataFrame(columns=["username", "email", "password", "role"])

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to add a user to the DataFrame and save to CSV
def add_user(username, email, password, role):
    hashed_password = hash_password(password)
    user_data.loc[len(user_data)+1] = [username, email, hashed_password, role]
    user_data.to_csv(csv_file_path, index=False)  # Save to CSV
    print("added")

# Function to check user credentials
def authenticate(username, password):
    user = user_data[user_data["username"] == username]
    if user.empty:
        return False
    return user.iloc[0]["password"] == hash_password(password)

def validate_email(email):
    pattern = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$" #tesQQ12@gmail.com
    if re.match(pattern, email):
        return True
    return False

def validate_username(username):
    pattern = "^[a-zA-Z0-9]*$"
    if re.match(pattern, username):
        return True
    return False

def authenticate(username, password):
    user = user_data[user_data["username"] == username]
    if user.empty:
        return False
    stored_password = user.iloc[0]["password"]
    if hash_password(password) == stored_password:
        return True
    return None

def get_role(username):
    user = user_data[user_data["username"] == username]
    return user.iloc[0]["role"]

def check_login_status():
    if st.session_state.username is not None:
        if st.sidebar.button("Log Out"):
            setupemptyprofile()
            st.success("Logged out successfully!")
    else:
        if st.sidebar.button("Log In"):
            switch_page("Home")

def setupemptyprofile():
    st.session_state.username = None
    st.session_state.role = None

def is_strong_password(password):
    # Minimum length check
    if len(password) < 8:
        return False
    # Check for uppercase, lowercase, digit, and special character
    if not re.search(r'[A-Z]', password) or \
       not re.search(r'[a-z]', password) or \
       not re.search(r'\d', password) or \
       not re.search(r'[!@#$%^&*]', password):
        return False
    return True

def user_update():
    action = st.selectbox("Select an action: ", ["Delete User", "Update User Email", "Update User Password"])
    if action == "Delete User":
        st.subheader("Delete User")
        search_string = st.text_input("Search user to delete", "")
        user_list = user_data["username"].tolist()
        filtered_users = [user for user in user_list if search_string in user]
        user_to_delete = st.selectbox("Select a user to delete:", sorted(filtered_users))
        if st.button("Delete User"):
            if user_to_delete:
                user_data.drop(user_data[user_data['username'] == user_to_delete].index, inplace=True)
                user_data.to_csv(csv_file_path, index=False)
                st.success(f"User '{user_to_delete}' has been deleted.")
            else:
                st.error("Please select a user to delete.")
    elif action == "Update User Email":
        st.subheader("Update User Details")
        search_string = st.text_input("Search user to update", "")
        user_list = user_data["username"].tolist()
        filtered_users = [user for user in user_list if search_string in user]
        user_to_update = st.selectbox("Select the user to update:", sorted(filtered_users))
        new_email = st.text_input("New Email for the user")
        user_index = user_data[user_data['username'] == user_to_update].index
        if st.button("Update"):
            if validate_email(new_email):
                user_data.loc[user_index, "email"] = new_email
                st.success("Email successfully changed")
            else:
                st.write("Invalid Email")
        user_data.to_csv(csv_file_path, index=False)
    elif action == "Update User Password":
        st.subheader("Change Password")
        search_string = st.text_input("Search user to update", "")
        user_list = user_data["username"].tolist()
        filtered_users = [user for user in user_list if search_string in user]
        user_to_update = st.selectbox("Select the user to update:", sorted(filtered_users))
        old_password = st.text_input("Old Password")
        new_password = st.text_input("New Password for the user")
        new_password2 = st.text_input("Confirm New Password")
        user_index = user_data[user_data['username'] == user_to_update].index
        if st.button("Update"):
            if is_strong_password(new_password2) and new_password == new_password2 and authenticate(user_to_update, old_password):
                user_data.loc[user_index, "password"] = hash_password(new_password2)
                st.success("Password successfully changed")
            else:
                st.write("Please check the password fields are correct!")
        user_data.to_csv(csv_file_path, index=False)
        
    
def sign_up():
    with st.form(key = "Create User", clear_on_submit=True):
        st.subheader(":green[Create User]")
        username = st.text_input(":blue[Username]")
        email = st.text_input(":blue[Email]")
        password = st.text_input(":blue[Password]", type="password")
        password2 = st.text_input(":blue[Confirm Password]", type="password")
        role = st.selectbox(":blue[Role]", ["admin", "user", "guest user"])
        btn1, btn2, btn3, btn4, btn5 = st.columns(5)
        with btn3:
            st.form_submit_button('Create')
    if email:
            if validate_email(email):
                if email not in user_data["email"].values:
                    if validate_username(username):
                        if username not in user_data["username"].values:
                            if len(username) >= 2:
                                if is_strong_password(password):
                                    if password == password2:
                                        # Add User to DB
                                        add_user(username, email, password2, role)
                                        st.success("User created successfully!")
                                        st.balloons()
                                    else:
                                        st.warning('Passwords Do Not Match')
                                else:
                                    st.warning('Please follow the password criteria to include at least one upper case letter, one lower case letter, a number and a special character')
                            else:
                                st.warning('Username Too short')
                        else:
                            st.warning('Username Already Exists')
                    else:
                        st.warning('Invalid Username')
                else:
                    st.warning('Email Already exists!!')
            else:
                st.warning('Invalid Email')

def generate_captcha():
    chrs = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
    txt = ""
    n = 4
    while (n):
        txt += chrs[random.randint(1, 1000) % 35]
        n -= 1
    captcha = ImageCaptcha()
    image = captcha.generate(txt)
    return txt, image

