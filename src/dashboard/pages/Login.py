import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page
import time

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, email TEXT, password TEXT)')

def add_userdata(username,email,password):
	c.execute('INSERT INTO userstable(username,email,password) VALUES (?,?,?)',(username,email,password))
	conn.commit()

def login_user(username,email,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND email = ? AND password = ?',(username,email,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def main():
	"""The login + signup page"""

	st.title("User Log In and Sign Up Page")

	menu = ["Login","SignUp"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")

	elif choice == "Login":
		st.subheader("Login Section")
		
		username = st.text_input("User Name")
		email = st.text_input("Email", type='default')
		password = st.text_input("Password",type='password')

		if st.button("Login"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,email,check_hashes(password,hashed_pswd))
			if result:
				username = st.empty()
				email = st.empty()
				password = st.empty()

				st.success("Logged In as {}".format(username))

				switch_page("conversations")
			else:
				username = st.empty()
				email = st.empty()
				password = st.empty()
				st.warning("Incorrect Username/Password")






	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_email= st.text_input("Email")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,new_email, make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



if __name__ == '__main__':
	main()