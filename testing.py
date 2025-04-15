import streamlit as st
import mysql.connector

#Dealing with the dataset
import pandas as pd

# Read the dataset (assuming it's in the same directory as your notebook)
df = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\streamlit\DSBI_dataset.csv')


#data cleaning 
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r'[^\w]', '', regex=True)

# Step 2: Drop 'Sr_No' column if it exists
if 'Sr_No' in df.columns:
    df.drop(columns=['Sr_No'], inplace=True)

# Step 3: Clean 'Amount_in_USD' column — remove commas, convert to float
if df['Amount_in_USD'].dtype == 'object':
    df['Amount_in_USD'] = df['Amount_in_USD'].str.replace(',', '')
    df['Amount_in_USD'] = pd.to_numeric(df['Amount_in_USD'], errors='coerce')

# Step 4: Fill missing values in key columns with default labels
df['City__Location'] = df['City__Location'].fillna("Unknown")
df['Industry_Vertical'] = df['Industry_Vertical'].fillna("Other")

# Step 5: Drop rows where 'Amount_in_USD' or 'Investors_Name' is missing
df = df.dropna(subset=['Amount_in_USD', 'Investors_Name'])

# Step 6: Convert 'Date_ddmmyyyy' to proper datetime format
df['Date_ddmmyyyy'] = pd.to_datetime(df['Date_ddmmyyyy'], dayfirst=True, errors='coerce')

# Step 7: Standardize text formatting in selected columns
text_cols = ['Startup_Name', 'Industry_Vertical', 'City__Location', 'Investors_Name']
for col in text_cols:
    df[col] = df[col].astype(str).str.title().str.strip()

# Step 8: If 'Profitable' column does not exist, add it with random 0s and 1s
import numpy as np
if 'Profitable' not in df.columns:
    np.random.seed(42)  # Optional: for reproducibility
    df['Profitable'] = np.random.randint(0, 2, size=len(df))


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Define target and features
X = df[['Amount_in_USD', 'Industry_Vertical', 'City__Location', 'InvestmentnType']]
y = df['Profitable']

# 2. Handle categorical columns
categorical_cols = ['Industry_Vertical', 'City__Location', 'InvestmentnType']
numeric_cols = ['Amount_in_USD']

# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

# 4. Build pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Fit model
clf.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = clf.predict(X_test)
# --- Database connection ---
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="Aryan@2004",
        database="Investor_DB"
    )

# --- Fetch user from DB ---
def fetch_user(username, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM users WHERE username = %s AND password = %s"
    cursor.execute(query, (username, password))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

# --- Example function to process input (replace with your own logic) ---
def predict_startup_profitability(amount,industry,city,investment_type ):
    # Take inputs from the user
    # Store input into a DataFrame (same structure as training)
    user_input = pd.DataFrame({
        'Amount_in_USD': [amount],
        'Industry_Vertical': [industry],
        'City__Location': [city],
        'InvestmentnType': [investment_type]
    })

    # Predict using the trained pipeline
    prediction = clf.predict(user_input)[0]
    probability = clf.predict_proba(user_input)[0][1]  # Probability of being class 1 (Profitable)

    # Output result
    if prediction == 1:
        return f"\n✅ This startup is likely to be PROFITABLE with a confidence of {probability*100:.2f}%"
    else:
        return f"\n❌ This startup is likely to NOT be profitable. Confidence of being profitable: {probability*100:.2f}%"

# --- Login Page ---
def login_page():
    st.title("💵 Investor Intelligence")
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
    st.title("📊 Investment Input Form")
    st.write(f"Logged in as **{st.session_state.username}**")

    with st.form("input_form"):
        amount = st.number_input("Amount in USD", min_value=1.0)
        vertical = st.text_input("Industry Vertical")
        city = st.text_input("City")
        investment_type = st.text_input("Investment Type")

        submitted = st.form_submit_button("Submit")

        if submitted:
            result = predict_startup_profitability(amount, vertical, city, investment_type)
            st.success("✅ Input Submitted Successfully!")
            st.info(f"📈 Output: {result}")

# --- App Controller ---
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        input_form_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
