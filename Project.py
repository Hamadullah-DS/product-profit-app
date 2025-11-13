import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Fiverr Spammer Detection System", layout="wide")
st.title("Fiverr Spammer Detection System")
st.write("Identify suspicious or scam clients on Fiverr using behavioral and text-based patterns.")

st.sidebar.header("Upload or Use Sample Data")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset", value=(uploaded_file is None))

if use_sample:
    np.random.seed(42)
    data = {
        "Contains_Link": np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
        "Contains_Outside_Contact": np.random.choice([0, 1], size=1000, p=[0.75, 0.25]),
        "Message_Length_Words": np.random.randint(5, 250, size=1000),
        "Poor_Grammar_Score": np.round(np.random.uniform(1, 10, size=1000), 1),
        "Account_Age_Days": np.random.randint(0, 3650, size=1000),
        "Num_Past_Orders": np.random.randint(0, 50, size=1000),
        "Contains_Urgency_Phrase": np.random.choice([0, 1], size=1000, p=[0.8, 0.2]),
        "Budget_USD": np.round(np.random.uniform(5, 2000, size=1000), 2),
        "Contains_Free_Work_Request": np.random.choice([0, 1], size=1000, p=[0.85, 0.15]),
        "Response_Time_Minutes": np.round(np.random.uniform(0.1, 1200, size=1000), 2),
    }
    df = pd.DataFrame(data)
    spam_probability = (
        0.3 * df["Contains_Link"]
        + 0.25 * df["Contains_Outside_Contact"]
        + 0.2 * (df["Poor_Grammar_Score"] < 4)
        + 0.15 * (df["Account_Age_Days"] < 30)
        + 0.2 * (df["Contains_Free_Work_Request"])
        + 0.15 * (df["Budget_USD"] < 20)
    )
    df["Spam_Label"] = (spam_probability > np.random.uniform(0, 1, size=1000)).astype(int)
    st.sidebar.info("Using sample dataset (1000 rows).")
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        st.warning("Please upload a CSV file or select sample dataset.")
        st.stop()

tab1, tab2, tab3 = st.tabs(["Data Preview", "Data Visualization", "Predict"])

with tab1:
    st.subheader("Data Overview")
    st.dataframe(df.head(10))
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.write("### Missing Values")
    st.write(df.isnull().sum())

with tab2:
    st.subheader("Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Spam vs Genuine Clients")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Spam_Label", data=df, palette="coolwarm", ax=ax1)
        ax1.set_xticklabels(["Genuine", "Spammer"])
        st.pyplot(fig1)

    with col2:
        st.write("#### Average Budget by Client Type")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x="Spam_Label", y="Budget_USD", data=df, palette="Set2", ax=ax2)
        ax2.set_xticklabels(["Genuine", "Spammer"])
        st.pyplot(fig2)

    st.write("#### Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

with tab3:
    st.subheader("Predict a New Client Message")
    col1, col2, col3 = st.columns(3)

    with col1:
        link = st.selectbox("Contains Link?", [0, 1])
        contact = st.selectbox("Contains Outside Contact?", [0, 1])
        urgency = st.selectbox("Contains Urgency Phrase?", [0, 1])

    with col2:
        grammar = st.slider("Grammar Score (1 = poor, 10 = excellent)", 1.0, 10.0, 6.0)
        account_age = st.number_input("Account Age (days)", 0, 3650, 120)
        past_orders = st.number_input("Number of Past Orders", 0, 100, 3)

    with col3:
        budget = st.number_input("Budget (USD)", 5.0, 5000.0, 100.0)
        free_work = st.selectbox("Asked for Free Work?", [0, 1])
        response = st.number_input("Response Time (minutes)", 0.1, 1440.0, 10.0)

    X = df.drop("Spam_Label", axis=1)
    y = df["Spam_Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Model Evaluation Results")

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(cr).transpose())

    st.write(f"### Model Accuracy: **{acc * 100:.2f}%**")

    input_data = pd.DataFrame({
        "Contains_Link": [link],
        "Contains_Outside_Contact": [contact],
        "Message_Length_Words": [np.random.randint(10, 200)],  # random placeholder
        "Poor_Grammar_Score": [grammar],
        "Account_Age_Days": [account_age],
        "Num_Past_Orders": [past_orders],
        "Contains_Urgency_Phrase": [urgency],
        "Budget_USD": [budget],
        "Contains_Free_Work_Request": [free_work],
        "Response_Time_Minutes": [response]
    })

    if st.button("Predict Spam Likelihood"):
        pred = model.predict(input_data)[0]
        if pred == 1:
            st.error("This client is likely a **Spammer**. Be cautious!")
        else:
            st.success("This client seems **Genuine** and safe to proceed with.")
