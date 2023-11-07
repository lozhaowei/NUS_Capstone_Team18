import pandas as pd
import streamlit as st
import hashlib
import os
import re
import random
import pymysql
import datetime

import extra_streamlit_components as stx
from streamlit_extras.switch_page_button import switch_page
from captcha.image import ImageCaptcha
from src.data.make_datasets import pull_raw_data
from src.dashboard.data.spark_pipeline import SparkPipeline
from decouple import config

# Define the path for the CSV file
csv_file_path = "datasets/final/user_data.csv"

CONN_PARAMS = {
    'host': config('DB_HOST'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'port': int(config('DB_PORT')),
    'database': config('DB_NAME'),
}

def insert_user_data(table_name, data):
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        # Convert 'username' column to string before insertion
        data['username'] = data['username'].astype(str)
        # Convert NaN values to None for proper insertion
        data = data.where(pd.notna(data), None)

        # Insert data only if the combination of 'dt' and 'model' is unique
        for _, row in data.iterrows():
            insert_query = f'''
            INSERT INTO {table_name} (`username`, `email`, `password`, `role`) 
            SELECT %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM {table_name} WHERE `username` = %s AND `email` = %s
            )
            '''
            cursor.execute(insert_query, (row['username'], row['email'], row['password'], row['role'], row['username'], row['email']))

        conn.commit()
        conn.close()

        print(f"Data updated in MySQL table '{table_name}' successfully.")

    except Exception as e:
        print("Error:", e)


# Create an empty DataFrame to store user details or load from CSV if it exists
#pull_raw_data(['nus_user_data'])
# If the table is not extracted/does not exist, use the local one instead, if local one also dont have, create a new table
#if the table exists on AWS server, then a feather file will be created in datasets/raw
if os.path.exists('datasets/raw/nus_user_data.feather'):
    #read the data
    user_data = pd.read_feather('datasets/raw/nus_user_data.feather')
    if os.path.exists(csv_file_path):
        user_data = pd.read_csv(csv_file_path)
#if table does not exist on AWS server, we will check if it exists on the final file(which is a local saved copy)
elif os.path.exists(csv_file_path):
    print("retrieving local user_data file")
    user_data = pd.read_csv(csv_file_path)
#if there isnt a file on both directory, then create a new empty table to store user data
else:
    print("new df")
    user_data = pd.DataFrame(columns=["username", "email", "password", "role"])

print("inserting data")
insert_user_data("nus_user_data", user_data)     

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to add a user to the DataFrame and save to CSV
def add_user(username, email, password, role):
    hashed_password = hash_password(password)
    user_data.loc[len(user_data)+1] = [username, email, hashed_password, role]
    user_data.to_csv(csv_file_path, index=False)  # Save to CSV
    insert_user_data("nus_user_data", user_data)
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

def get_password(username):
    user = user_data[user_data["username"] == username]
    return user.iloc[0]["password"]

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
    action = st.selectbox("Select an action: ", ["Delete User", "Update User Email", "Update User Password", "Update User Role"])
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
                insert_user_data("nus_user_data", user_data)
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
        insert_user_data("nus_user_data", user_data)
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
        insert_user_data("nus_user_data", user_data)
    elif action == "Update User Role":
        st.subheader("Update User Role")
        search_string = st.text_input("Search user to update", "")
        user_list = user_data["username"].tolist()
        filtered_users = [user for user in user_list if search_string in user]
        user_to_update = st.selectbox("Select the user to update:", sorted(filtered_users))
        user_index = user_data[user_data['username'] == user_to_update].index
        new_role = st.selectbox("Role", ["admin", "user", "guest user"])
        if st.button("Update"):
            user_data.loc[user_index, "role"] = new_role
            st.success("User role successfully changed")
        user_data.to_csv(csv_file_path, index=False)
        insert_user_data("nus_user_data", user_data)
        
    
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

import base64

# Pseudo-code for generating a session token
def generate_session_token(username, role):
    # Combine username and role as a string
    user_info = f"{username}:{role}"
    # Encode the user information to base64
    encoded_user_info = base64.b64encode(user_info.encode()).decode()
    # Create and return the session token
    return encoded_user_info

# Pseudo-code for extracting username and role from a session token
def get_username_from_session_token(session_token):
    try:
        # Decode base64 and split the user info
        decoded_user_info = base64.b64decode(session_token.encode()).decode()
        username, role = decoded_user_info.split(":")
        return username
    except Exception as e:
        return None

def get_role_from_session_token(session_token):
    try:
        # Decode base64 and split the user info
        decoded_user_info = base64.b64decode(session_token.encode()).decode()
        username, role = decoded_user_info.split(":")
        return role
    except Exception as e:
        return None

@st.cache_data()

def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

def login_with_remember_me():
    username = st.text_input("Username", key="unique username")
    password = st.text_input("Password", type="password")
    remember_me = st.checkbox(":green[Remember Me]", key="unique remember me")
    
    if st.button("✅Login"):
        if remember_me:
            cookie_manager.set("username", username, expires_at=(datetime.date.today() + datetime.timedelta(days=7)))
            #cookie_manager.set(username, username, expires_at=datetime.datetime(year=2023, month=11, day=3))

        #print(captcha, original_text, captcha_image)
        #if captcha == original_text:
        
        if authenticate(username, password):
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
        else:
            st.error("Login failed. Please check your credentials and/or your CAPTCHA.")
