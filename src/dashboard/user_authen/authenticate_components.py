import pandas as pd
import streamlit as st
import hashlib
import os
import re
import random
import pymysql
import datetime
import base64

import extra_streamlit_components as stx
from streamlit_extras.switch_page_button import switch_page
from captcha.image import ImageCaptcha
from src.data.make_datasets import pull_raw_data
from src.dashboard.data.spark_pipeline import SparkPipeline
from decouple import config

# This pages comprises of the various components of user authentication that will be used for our user login and management process

# Define the path for the CSV file that stores user data
csv_file_path = "datasets/final/user_data.csv"
user_data_table = "nus_user_data"
user_roles = ["admin", "user_for_both", "user_for_conversation", "user_for_videos", "user"]

# Extracting the parameters for connecting to the SQL tables
CONN_PARAMS = {
    'host': config('DB_HOST'),
    'user': config('DB_USER'),
    'password': config('DB_PASSWORD'),
    'port': int(config('DB_PORT')),
    'database': config('DB_NAME'),
}

def check_table_exist(table_name):
    """
    Checks if a table exists in the database, and creates the table if it doesn't.

    Parameters:
    - table_name (str): The name of the table to check.

    Returns:
    - bool: True if the table exists, False if it doesn't.
    """
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        # Query to see if the table exist in the database
        table_exists_query = f"SHOW TABLES LIKE '{table_name}'"
        table_exists = cursor.execute(table_exists_query)
        # If the table does not exist, we create the table in the AWS server
        
        if not table_exists:
            print("table does not exist in AWS, creating new table")
            columns = {
                'username': 'VARCHAR(45)',
                'email': 'VARCHAR(45)',
                'password': 'VARCHAR(255)',
                'role': 'VARCHAR(45)'
            }
            create_table_query = f"CREATE TABLE {table_name} ({', '.join([f'{col} {datatype}' for col, datatype in columns.items()])}, PRIMARY KEY (username))"
            cursor.execute(create_table_query)
            conn.close()
            return False
        else:
            conn.close()
            return True

    except Exception as e:
        print("Error:", e)

# This function inserts the local user data credential table to the AWS server
def insert_user_data(table_name, data):
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        # Convert 'username' column to string before insertion
        data['username'] = data['username'].astype(str)
        # Convert NaN values to None for proper insertion
        data = data.where(pd.notna(data), None)

        # Insert data only if the combination of 'username' and 'email' is unique
        for _, row in data.iterrows():
            insert_query = f'''
            INSERT INTO {table_name} (`username`, `email`, `password`, `role`) 
            SELECT %s, %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM {table_name} WHERE `username` = %s
            )
            '''
            cursor.execute(insert_query, (row['username'], row['email'], row['password'], row['role'], row['username']))
        
        conn.commit()
        conn.close()

        print(f"Data inserted in MySQL table '{table_name}' successfully.")

    except Exception as e:
        print("Error:", e)

def update_table(table_name, data):
    try:
        conn = pymysql.connect(**CONN_PARAMS)
        cursor = conn.cursor()

        for _, row in data.iterrows():
            username = row['username']
            email = row['email']
            password = row['password']
            role = row['role']

            query = f"SELECT COUNT(*) FROM {table_name} WHERE username = '{username}'"
            cursor.execute(query)
            record_exists = cursor.fetchone()[0]
            
            print("checking records")
            if record_exists:
                # Update the existing record
                update_query = f"UPDATE {table_name} SET email = '{email}', password = '{password}', role = '{role}' WHERE username = '{username}'"
                cursor.execute(update_query)
            else:
                # Insert a new record
                insert_query = f"INSERT INTO {table_name} (username, email, password, role) VALUES ('{username}', '{email}', '{password}', '{role}')"
                cursor.execute(insert_query)

        conn.commit()
        cursor.close()
        conn.close()

        print(f"Data updated in MySQL table '{table_name}' successfully.")

    except Exception as e:
        print("Error:", e)

#if the table exists on AWS server, then a feather file will be created in datasets/raw
if check_table_exist(user_data_table): 
    print("getting data from AWS")
    pull_raw_data([user_data_table])
    user_data = pd.read_feather('datasets/raw/nus_user_data.feather')
    user_data.to_csv(csv_file_path, index=False)
    user_data = pd.read_csv(csv_file_path)
    print(user_data)
else:
    #if table does not exist on AWS server, we will check if it exists on the final file(which is a locally saved copy)
    if os.path.exists(csv_file_path):
        print("retrieving local user_data file")
        user_data = pd.read_csv(csv_file_path)
    else:
        print("new df")
        #if there isnt a file on both directory, then create a new empty table to store user data
        user_data = pd.DataFrame(columns=["username", "email", "password", "role"])
#if there is already a local user data table, then we will insert/update the AWS table

#insert_user_data("nus_user_data", user_data)     

def get_user_data():
    return pd.read_csv(csv_file_path)

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to add a user to the DataFrame and save to CSV
def add_user(username, email, password, role):
    user_data = pd.read_csv(csv_file_path)
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

# Function to validate that email follow the correct pattern
def validate_email(email):
    pattern = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$" #tesQQ12@gmail.com
    if re.match(pattern, email):
        return True
    return False

# Ensure that the username field is valid
def validate_username(username):
    pattern = "^[a-zA-Z0-9]*$"
    if re.match(pattern, username):
        return True
    return False

# Function to authenticate the user account based on matching credentials in the user_data_table
def authenticate(username, password):
    user = user_data[user_data["username"] == username]
    if user.empty:
        return False
    stored_password = user.iloc[0]["password"]
    if hash_password(password) == stored_password:
        return True
    return None

# Function to obtain the role of the user
def get_role(username):
    user = user_data[user_data["username"] == username]
    return user.iloc[0]["role"]

# Initialise/Reset the session state
def setupemptyprofile():
    st.session_state.username = None
    st.session_state.role = None

# Function to check the login status of the user, if the user is logged in, there will be a "Logout" button at the side bar,
# if the user has not logged in, there will be a "Log In" button at the side bar, clicking on this button directs the user to Home page to log in.
def check_login_status():
    if 'username' not in st.session_state:
        set_up_empty_profile()
    elif st.session_state.username is not None:
        if st.sidebar.button("Log Out"):
            set_up_empty_profile()
            st.success("Logged out successfully!")
    else:
        if st.sidebar.button("Log In"):
            switch_page("Home")

def set_up_empty_profile():
    st.session_state.username = None
    st.session_state.role = None

# This function performs strong password validation such that it must include an upper case letter, a lower case letter
# a number and a special character
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

# This function allows the different credential of users to be updated
def user_update():
    user_data = pd.read_csv(csv_file_path)
    # The selectbox allows the admins to select different operations
    # All the changes will be reflected in the user data table in AWS Server
    action = st.selectbox("Select an action: ", ["Delete User", "Update User Email", "Update User Password", "Update User Role"])
    # When deleting user, the admin will search for a user by typing the username in the search bar
    # Then the admin can choose a user from the select box generated from the search result
    # The admin can then click on the button to delete the user
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
                #user_data = pd.read_csv(csv_file_path)
                #print(user_data)
                update_table("nus_user_data", user_data)
                st.success(f"User '{user_to_delete}' has been deleted.")
            else:
                st.error("Please select a user to delete.")

    # When updating user email, the admin will search for a user by typing the username in the search bar
    # Then the admin can choose a user from the select box generated from the search result
    # The admin can then enter a new email for the user and click on the button to update the email
    # There is also a check in place to validate the email
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
                user_data.to_csv(csv_file_path, index=False)
                #user_data = pd.read_csv(csv_file_path)
                update_table("nus_user_data", user_data)
                st.success("Email successfully changed")
            else:
                st.write("Invalid Email")
        
    # When updating user password, the admin will search for a user by typing the username in the search bar
    # Then the admin can choose a user from the select box generated from the search result
    # The admin can then enter a new password and confirm the password for the user and click on the button to update the password
    # There is also a check in place to validate whether the password is a strong password
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
                user_data.to_csv(csv_file_path, index=False)
                #user_data = pd.read_csv(csv_file_path)
                update_table("nus_user_data", user_data)
                st.success("Password successfully changed")
            else:
                st.write("Please check the password fields are correct!")


    # When updating user role, the admin will search for a user by typing the username in the search bar
    # Then the admin can choose a user from the select box generated from the search result
    # The admin can then select a new role and click on the button to update the roled
    elif action == "Update User Role":
        st.subheader("Update User Role")
        search_string = st.text_input("Search user to update", "")
        user_list = user_data["username"].tolist()
        filtered_users = [user for user in user_list if search_string in user]
        user_to_update = st.selectbox("Select the user to update:", sorted(filtered_users))
        user_index = user_data[user_data['username'] == user_to_update].index
        new_role = st.selectbox("Role", user_roles)
        if st.button("Update"):
            user_data.loc[user_index, "role"] = new_role
            
            user_data.to_csv(csv_file_path, index=False)
            #user_data = pd.read_csv(csv_file_path)
            update_table("nus_user_data", user_data)
            st.success("User role successfully changed")

# This function allows users to change their own password (it is meant for the non-admin users)
def change_password():
    st.subheader("Change Password")
    user_to_update = st.session_state.username
    old_password = st.text_input("Old Password")
    new_password = st.text_input("New Password")
    new_password2 = st.text_input("Confirm New Password")
    user_index = user_data[user_data['username'] == user_to_update].index
    if st.button("Update"):
        if is_strong_password(new_password2) and new_password == new_password2 and authenticate(user_to_update, old_password):
            user_data.loc[user_index, "password"] = hash_password(new_password2)
            st.success("Password successfully changed")
        else:
            st.write("Please check the password fields are correct!")
    user_data.to_csv(csv_file_path, index=False)
    #user_data = pd.read_csv(csv_file_path)
    update_table("nus_user_data", user_data)

# This function creates the sign up form for the admin to create account for new users
# The various vailidation for username, email and password will still hold.
def sign_up():
    with st.form(key = "Create User", clear_on_submit=True):
        st.subheader(":green[Create User]")
        username = st.text_input(":blue[Username]")
        email = st.text_input(":blue[Email]")
        password = st.text_input(":blue[Password]", type="password")
        password2 = st.text_input(":blue[Confirm Password]", type="password")
        role = st.selectbox(":blue[Role]", user_roles)
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

#@st.cache_data()

def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

def login_with_remember_me():
    username = st.text_input("Username", key="unique username")
    password = st.text_input("Password", type="password")
    remember_me = st.checkbox(":green[Remember Me]", key="unique remember me")
    
    if st.button("Login"):
        if remember_me:
            if authenticate(username, password):
                cookie_manager.set("username", username, expires_at=(datetime.date.today() + datetime.timedelta(days=7)))

        if authenticate(username, password):
            st.session_state.username = username
            st.session_state.role = get_role(username)
            st.success("Login successful!")
            st.text("Welcome! You can now navigate through the different pages")
        else:
            st.error("Login failed. Please check your credentials.")
