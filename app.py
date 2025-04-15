import streamlit as st
import mysql.connector

# Function to connect to MySQL
def get_connection():
    return mysql.connector.connect(
        host="localhost",             # ✅ host only
        port=3306,                    # ✅ port separately
        user="root",                 # 👈 Your MySQL username
        password="Aryan@2004",       # 👈 Your MySQL password
        database="investor_db"       # 👈 Your database name
    )

# Function to fetch user from DB using username & password
def fetch_user(username, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM users WHERE username = %s AND password = %s"
    cursor.execute(query, (username, password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

# Streamlit UI
st.title("🔐 Login Page")

username = st.text_input("Username")
password = st.text_input("Password", type="password")
login_button = st.button("Login")

if login_button:
    if username and password:
        user = fetch_user(username, password)
        if user:
            st.success(f"Welcome, {user['username']}! 🎉")
            # You can load your main app or dataset here
        else:
            st.error("Invalid username or password")
    else:
        st.warning("Please enter both username and password")
