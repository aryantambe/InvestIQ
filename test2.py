import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Dark Theme Setup ---
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #ffffff;
        }
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        input, textarea, select {
            background-color: #1e2128 !important;
            color: white !important;
        }
        .stButton>button {
            background-color: #6c63ff;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.5em;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #a393eb;
        }
        .stTextInput>div>div>input {
            background-color: #1e2128 !important;
            color: white !important;
        }
        .stNumberInput>div>div>input {
            background-color: #1e2128 !important;
            color: white !important;
        }
        .title-style {
            text-align: center;
            font-size: 2.5em;
            color: #6c63ff;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .section-title {
            font-size: 1.4em;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess the dataset
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\streamlit\DSBI_dataset.csv')
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r'[^\w]', '', regex=True)
if 'Sr_No' in df.columns:
    df.drop(columns=['Sr_No'], inplace=True)
if df['Amount_in_USD'].dtype == 'object':
    df['Amount_in_USD'] = df['Amount_in_USD'].str.replace(',', '')
    df['Amount_in_USD'] = pd.to_numeric(df['Amount_in_USD'], errors='coerce')
df['City__Location'] = df['City__Location'].fillna("Unknown")
df['Industry_Vertical'] = df['Industry_Vertical'].fillna("Other")
df = df.dropna(subset=['Amount_in_USD', 'Investors_Name'])
df['Date_ddmmyyyy'] = pd.to_datetime(df['Date_ddmmyyyy'], dayfirst=True, errors='coerce')
text_cols = ['Startup_Name', 'Industry_Vertical', 'City__Location', 'Investors_Name']
for col in text_cols:
    df[col] = df[col].astype(str).str.title().str.strip()
if 'Profitable' not in df.columns:
    np.random.seed(42)
    df['Profitable'] = np.random.randint(0, 2, size=len(df))

# ML setup
X = df[['Amount_in_USD', 'Industry_Vertical', 'City__Location', 'InvestmentnType']]
y = df['Profitable']
categorical_cols = ['Industry_Vertical', 'City__Location', 'InvestmentnType']
numeric_cols = ['Amount_in_USD']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# --- DB ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="Aryan@2004",
        database="Investor_DB"
    )

def fetch_user(username, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def predict_startup_profitability(amount, industry, city, investment_type):
    user_input = pd.DataFrame({
        'Amount_in_USD': [amount],
        'Industry_Vertical': [industry],
        'City__Location': [city],
        'InvestmentnType': [investment_type]
    })
    prediction = clf.predict(user_input)[0]
    probability = clf.predict_proba(user_input)[0][1]
    if prediction == 1:
        return f"✅ This startup is likely to be PROFITABLE\n💡 Confidence: {probability*100:.2f}%"
    else:
        return f"❌ This startup is likely to NOT be profitable\n📉 Confidence of profitability: {probability*100:.2f}%"

# --- Login Page ---
def login_page():
    st.markdown('<div class="title-style">💼 Investor Intelligence Login</div>', unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            user = fetch_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user['username']
                st.success(f"Welcome, {user['username']}! 🎉")
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")

# --- Input Form Page ---
def input_form_page():
    st.markdown('<div class="title-style">📊 Investment Input Form</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>👤 Logged in as: <b>{st.session_state.username}</b></div><br>", unsafe_allow_html=True)

    with st.form("input_form"):
        amount = st.number_input("💰 Amount in USD", min_value=1.0)
        vertical = st.text_input("🏢 Industry Vertical")
        city = st.text_input("🌆 City")
        investment_type = st.text_input("📄 Investment Type")
        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            result = predict_startup_profitability(amount, vertical, city, investment_type)
            st.success("✅ Input Submitted Successfully!")
            st.info(result)

# --- Main Controller ---
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        input_form_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
