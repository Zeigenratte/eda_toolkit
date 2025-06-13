import pandas as pd
import streamlit as st
import os

def load_data():
    st.sidebar.header("Step 1: Upload or Use Default Data")

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Upload your own dataset
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Custom dataset uploaded.")
        source = "User Upload"
    else:
        # Use default dataset
        default_path = os.path.join("data", "cirrhosis.csv")
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
            st.sidebar.info("Using default dataset: `cirrhosis.csv`")
            source = "Default Dataset"
        else:
            st.error("❌ No data file found. Please upload a CSV.")
            return None, None

    return df, source
