import pandas as pd
import streamlit as st

class OverviewAnalysis:
    """
    A class to conduct quick exploratory data analysis (EDA) on a given DataFrame. 
    Handels:
        - A preview of the data (head and tail, transposed for readability)
        - Descriptive statistics for both numeric and categorical variables
        - Summary of missing values (counts and percentages)
        - Summary of duplicate rows
        - A comprehensive EDA report (excluding duplicates)

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to analyze.
    name : str, optional
        A name for the dataset (default is "Dataset").
    """

    def __init__(self, df: pd.DataFrame, name: str = "Dataset"):
        self.df = df
        self.name = name

    def data_preview(self) -> None:
        """
        Displays the shape of the dataset and previews the first and last few rows, 
        transposed for clarity.
        """
        st.subheader(f"Shape of '{self.name}'")
        st.write(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")

        st.subheader(f"Preview of '{self.name}' - Head (Top 5 Rows)")
        st.dataframe(self.df.head().T)

        st.subheader(f"Preview of '{self.name}' - Tail (Last 5 Rows)")
        st.dataframe(self.df.tail().T)

    def summarize_overview(self) -> None:
        """
        Displays combined descriptive statistics for both numeric and categorical columns.
        Includes dtype and unique value count for all columns.
        """
        st.subheader(f"Combined Data Summary for '{self.name}'")

        # Basic type and uniqueness summary
        summary = pd.DataFrame({
            "dtype": self.df.dtypes,
            "unique": self.df.nunique()
        })

        # Descriptive stats
        numeric_stats = self.df.describe().T
        categorical_stats = self.df.describe(include='object').T

        # Remove duplicate 'unique' columns
        if 'unique' in numeric_stats.columns:
            numeric_stats = numeric_stats.drop(columns=['unique'])
        if 'unique' in categorical_stats.columns:
            categorical_stats = categorical_stats.drop(columns=['unique'])

        # Concatenate both
        full_stats = pd.concat([categorical_stats, numeric_stats], axis=0)
        summary = summary.join(full_stats, how='left')

        st.dataframe(summary)

    def missing_values(self) -> None:
        """
        Displays the number and percentage of missing values per column and overall.
        """
        total_missing = self.df.isnull().sum().sum()
        missing_percent = round((total_missing / self.df.size) * 100, 1)

        st.subheader(f"Missing Value Summary for '{self.name}'")

        st.write("Missing values per column:")
        st.dataframe(self.df.isnull().sum().to_frame(name='Missing Values'))

        st.markdown(f"**Total Missing Values:** {total_missing}")
        st.markdown(f"**Percentage of Total Entries Missing:** {missing_percent}%")


    def duplicates_summary(self) -> None:
        """
        Displays the number and percentage of duplicate rows. 
        Also outputs the duplicate rows if present.
        """
        total_duplicates = self.df.duplicated().sum()
        duplicate_percent = round((total_duplicates / len(self.df)) * 100, 1)

        st.subheader(f"Duplicate Row Summary for '{self.name}'")
        st.markdown(f"**Total Duplicates:** {total_duplicates}")
        st.markdown(f"**Percentage of Duplicates:** {duplicate_percent}%")

        if total_duplicates > 0:
            st.write("Duplicate rows (all columns):")
            st.dataframe(self.df[self.df.duplicated()])


    def report(self) -> None:
        """
        Runs a full EDA sequence including:
            - Data preview
            - Descriptive summary
            - Missing value overview

        Note: This method excludes duplicate row summary to keep reports concise.
        """
        st.header(f"Full Overview Report for '{self.name}'")
        self.data_preview()
        self.summarize_overview()
        self.missing_values()
