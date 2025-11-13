import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Page Setup"
# -----------------------------
st.set_page_config(page_title="Product Profitability Dashboard", layout="wide")
st.title("📊 Product Profitability Dashboard")
st.write("Analyze and predict product profitability with interactive visuals and ML predictions.")

# -----------------------------
# Sidebar - Upload or Sample
# -----------------------------
st.sidebar.header("1️⃣ Upload Data or Use Sample")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset", value=(uploaded_file is None))

if use_sample:
    np.random.seed(42)
    categories = ['Electronics', 'Apparel', 'Home & Kitchen', 'Books', 'Sports']
    data = {
        'Product_ID': [f'P{1000 + i}' for i in range(1000)],
        'Category': np.random.choice(categories, 1000),
        'Sales_Units': np.random.randint(1, 500, 1000),
        'Unit_Price': np.round(np.random.uniform(10, 500, 1000), 2),
        'Cost_Per_Unit': np.round(np.random.uniform(5, 400, 1000), 2),
        'Date_Added': pd.to_datetime(np.random.randint(
            pd.Timestamp('2018-01-01').value // 10 ** 9,
            pd.Timestamp('2025-01-01').value // 10 ** 9,
            1000
        ), unit='s')
    }
    df = pd.DataFrame(data)
    df['Total_Revenue'] = df['Sales_Units'] * df['Unit_Price']
    df['Total_Cost'] = df['Sales_Units'] * df['Cost_Per_Unit']
    df['Profit'] = df['Total_Revenue'] - df['Total_Cost']
    df['Profit_Margin'] = df['Profit'] / df['Total_Revenue']
    st.sidebar.info("Using sample dataset with 1000 rows.")
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Date_Added'])
        st.sidebar.success("File uploaded successfully.")
    else:
        st.warning("Please upload a dataset or enable sample dataset.")
        st.stop()

# -----------------------------
# Prepare Model Data
# -----------------------------
X = pd.get_dummies(df[['Category', 'Sales_Units', 'Unit_Price', 'Cost_Per_Unit']], drop_first=True)
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Data Visualization", "🤖 Predict Profit"])

# -----------------------------
# Tab 1: Data Preview
# -----------------------------
with tab1:
    st.subheader("🔍 Dataset Overview")
    st.dataframe(df.head(10))
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    st.write("### Model Performance on Test Data")
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

# -----------------------------
# Tab 2: Data Visualization
# -----------------------------
with tab2:
    st.subheader("📈 Visual Insights")

    category_filter = st.multiselect("Filter by Category", options=df['Category'].unique(),
                                     default=df['Category'].unique())
    date_range = st.date_input("Select Date Range", [df['Date_Added'].min(), df['Date_Added'].max()])

    df_filtered = df[(df['Category'].isin(category_filter)) &
                     (df['Date_Added'] >= pd.to_datetime(date_range[0])) &
                     (df['Date_Added'] <= pd.to_datetime(date_range[1]))]

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Top 10 Products by Profit")
        top_profit = df_filtered.sort_values('Profit', ascending=False).head(10)
        fig1, ax1 = plt.subplots()
        sns.barplot(x='Profit', y='Product_ID', data=top_profit, palette='Greens_r', ax=ax1)
        ax1.set_xlabel('Profit ($)')
        ax1.set_ylabel('Product ID')
        st.pyplot(fig1)

    with col2:
        st.write("#### Profit vs Sales Units")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='Sales_Units', y='Profit', hue='Category', data=df_filtered, palette='Set2', ax=ax2)
        ax2.set_xlabel('Sales Units')
        ax2.set_ylabel('Profit ($)')
        st.pyplot(fig2)

    st.write("#### Category-wise Average Profit & Profit Margin")
    cat_summary = df_filtered.groupby('Category')[['Profit', 'Profit_Margin']].mean().reset_index()
    fig3, ax3 = plt.subplots()
    cat_summary.plot(x='Category', y=['Profit', 'Profit_Margin'], kind='bar', ax=ax3, color=['#1f77b4', '#ff7f0e'])
    ax3.set_ylabel('Value')
    st.pyplot(fig3)

    st.write("#### Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_filtered[['Sales_Units', 'Unit_Price', 'Cost_Per_Unit', 'Total_Revenue', 'Total_Cost', 'Profit',
                             'Profit_Margin']].corr(),
                annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

# -----------------------------
# Tab 3: Predict Profit
# -----------------------------
with tab3:
    st.subheader("💬 Predict Product Profit")

    col1, col2 = st.columns(2)

    with col1:
        category_input = st.selectbox("Category", df['Category'].unique())
        sales_input = st.number_input("Sales Units", 1, 10000, 100)
        unit_price_input = st.number_input("Unit Price ($)", 1.0, 5000.0, 50.0)

    with col2:
        cost_input = st.number_input("Cost per Unit ($)", 0.1, 4000.0, 30.0)

    # Prepare input for prediction
    input_df = pd.DataFrame({'Category': [category_input],
                             'Sales_Units': [sales_input],
                             'Unit_Price': [unit_price_input],
                             'Cost_Per_Unit': [cost_input]})
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    input_df_encoded = input_df_encoded.reindex(columns=X.columns, fill_value=0)

    if st.button("🚀 Predict Profit"):
        pred_profit = model.predict(input_df_encoded)[0]
        total_revenue = sales_input * unit_price_input
        profit_margin = pred_profit / total_revenue if total_revenue != 0 else 0

        st.metric(label="Estimated Profit ($)", value=f"{pred_profit:.2f}")
        st.metric(label="Estimated Profit Margin (%)", value=f"{profit_margin * 100:.2f}")

        if pred_profit > 0:
            st.success("✅ Product is Profitable")
        else:
            st.error("⚠️ Product is Underperforming")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
**About this Dashboard:**
Professional Product Profitability Analysis with interactive visuals and ML-based profit predictions. Built with Streamlit, Pandas, Numpy, Matplotlib, Seaborn, and Scikit-learn.
""")
