import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from model_pipline import arrangeData

st.set_page_config(page_title="EDA - Car Price Prediction", layout="wide")

st.title("🔎 Exploratory Data Analysis (EDA)")

# -----------------------------
# Get dataset from session_state or let user upload
# -----------------------------
if "train_data" in st.session_state:
    df = st.session_state["train_data"]
    st.success("✅ Using dataset from main app (uploaded in training).")
else:
    eda_file = st.file_uploader("Upload dataset for EDA (CSV)", type=["csv"], key="eda")
    if eda_file:
        import pandas as pd
        df = pd.read_csv(eda_file)
        df = arrangeData(df)
        st.info("Using separately uploaded dataset.")
    else:
        st.warning("Please upload data in the main app or here to explore.")
        st.stop()

st.write("### Data Preview")
st.dataframe(df.head())

# -----------------------------
# Tabs for different EDA sections
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Summary",
    "📈 Distributions",
    "📉 Outliers",
    "🔎 Categorical Analysis"
])

with tab1:
    st.subheader("Dataset Summary")
    st.write(df.describe(include="all").transpose())

with tab2:
    st.subheader("Distributions of Numerical Features")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        selected_num = st.selectbox("Select a numeric column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_num}")
        st.pyplot(fig)
        st.write(f"""
        {selected_num} Analysis: Mean {selected_num}: {df[selected_num].describe().mean()}

        Median {selected_num}: {df[selected_num].describe().median()}

        {selected_num} Range: {df[selected_num].describe().min()}  –  {df[selected_num].describe().max()}

        75% of cars are {selected_num} below {df[selected_num].describe()['75%']}

        """)
    else:
        st.warning("No numeric columns found.")

with tab3:
    st.subheader("Outlier Detection")

    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['price'] < Q1 - 1.5 * IQR) | (df['price'] > Q3 + 1.5 * IQR)]
    st.dataframe(outliers.head())

with tab4:
    st.subheader("Categorical Value Counts")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        selected_cat = st.selectbox("Select a categorical column", cat_cols)
        st.write(df[selected_cat].value_counts())

        fig, ax = plt.subplots()
        sns.countplot(y=selected_cat, data=df, order=df[selected_cat].value_counts().index, ax=ax)
        ax.set_title(f"Count Plot of {selected_cat}")
        st.pyplot(fig)
    else:
        st.warning("No categorical columns found.")
