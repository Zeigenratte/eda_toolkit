import streamlit as st
from cleaning import ImputationHandler, OutliersHandler

# Define the list of available cleaning and outlier detection methods
available_methods = [
    "Mean/Median Imputation",
    "MICE Imputation",
    "KNN Imputation",
    "Mode Imputation",
    "Random Hot Deck Imputation",
    "Listwise Deletion",
    "Pairwise Deletion",
    "Modified Z-Score (Outliers)",
    "Z-Score (Outliers)",
    "IQR (Outliers)",
    "Log IQR (Outliers)"
]

# Define columns available for cleaning or outlier detection
available_columns = [
    'N_Days',
    'Status',
    'Drug',
    'Age',
    'Sex',
    'Ascites',
    'Hepatomegaly',
    'Spiders',
    'Edema',
    'Bilirubin',
    'Cholesterol',
    'Albumin',
    'Copper',
    'Alk_Phos',
    'SGOT',
    'Tryglicerides',
    'Platelets',
    'Prothrombin',
    'Stage'
]

def apply_method(df, method, columns, subgroup, threshold, max_iter, n_neighbors):
    """
    Apply the specified cleaning or outlier detection method on the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input data.
    - method (str): Name of the cleaning/outlier method to apply.
    - columns (list): List of columns to target.
    - subgroup (str): Subgroup specification ("All", "Randomized", "Non Randomized").
    - threshold (float): Threshold value for outlier methods.
    - max_iter (int): Maximum iterations for MI imputation.
    - n_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
    - pd.DataFrame: Cleaned or processed DataFrame.
    """
    # Define subgroup filter expression based on subgroup argument
    if subgroup == "Non Randomized":
        subgroup_expression = 'Drug == "No Drug"'
    elif subgroup == "Randomized":
        subgroup_expression = 'Drug != "No Drug"'
    elif subgroup == "All":
        subgroup_expression = None
    else:
        subgroup_expression = None  # Default fallback

    # Dispatch method call to appropriate handler
    if method == "Mean/Median Imputation":
        return ImputationHandler(df).mean_median_imputation(
            columns=columns, subgroup=subgroup_expression
        )
    elif method == "MICE Imputation":
        return ImputationHandler(df).multiple_imputation(
            columns=columns, max_iter=max_iter, subgroup=subgroup_expression
        )
    elif method == "KNN Imputation":
        return ImputationHandler(df).knn_imputation(
            columns=columns, n_neighbors=n_neighbors, subgroup=subgroup_expression
        )
    elif method == "Mode Imputation":
        return ImputationHandler(df).mode_imputation(
            columns=columns, subgroup=subgroup_expression
        )
    elif method == "Random Hot Deck Imputation":
        return ImputationHandler(df).random_hot_deck_imputation(
            columns=columns, subgroup=subgroup_expression
        )
    elif method == "Listwise Deletion":
        return ImputationHandler(df).listwise_deletion(
            subgroup=subgroup_expression
        )
    elif method == "Pairwise Deletion":
        return ImputationHandler(df).pairwise_deletion(
            columns=columns, subgroup=subgroup_expression
        )
    elif method == "Modified Z-Score (Outliers)":
        return OutliersHandler(df).modified_z_score_method(
            columns=columns, threshold=threshold, remove=True, subgroup=subgroup_expression
        )
    elif method == "Z-Score (Outliers)":
        return OutliersHandler(df).z_score_method(
            columns=columns, threshold=threshold, remove=True, subgroup=subgroup_expression
        )
    elif method == "IQR (Outliers)":
        return OutliersHandler(df).iqr_method(
            columns=columns, threshold=threshold, remove=True, subgroup=subgroup_expression
        )
    elif method == "Log IQR (Outliers)":
        return OutliersHandler(df).log_iqr_method(
            columns=columns, threshold=threshold, remove=True, subgroup=subgroup_expression
        )
    else:
        # If the method is not recognized, return the original DataFrame unchanged
        return df


def cleaning_ui(label, df_key):
    """
    Build the Streamlit UI for selecting cleaning methods and their configurations.

    Parameters:
    - label (str): Label to identify the DataFrame instance in the UI.
    - df_key (str): Unique key suffix to manage Streamlit widget states.

    Returns:
    - List of tuples: Each tuple contains
      (method, columns, subgroup, threshold, max_iter, n_neighbors)
    """
    st.markdown(f"#### DataFrame {label}")

    # Allow multiple methods to be selected
    methods = st.multiselect(
        f"Select cleaning steps for DataFrame {label}",
        available_methods,
        key=f"methods_{df_key}"
    )

    config = []
    for i, method in enumerate(methods):
        st.markdown(f"**{method} (DataFrame {label})**")

        # Allow selecting columns, with an option for "All"
        columns = st.multiselect(
            f"Select columns for {method} ({label})",
            ["All"] + available_columns,
            key=f"columns_{df_key}_{i}"
        )
        # Interpret "All" selection as all available columns except the "All" placeholder
        if "All" in columns:
            columns = [col for col in available_columns]

        # Select subgroup for cleaning application
        subgroup = st.selectbox(
            f"Subgroup for {method} ({label})",
            ["All", "Non Randomized", "Randomized"],
            key=f"subgroup_{df_key}_{i}"
        )

        # Slider for outlier threshold (meaningful only for outlier methods)
        threshold = st.slider(
            f"Threshold (for Outliers) ({label})",
            min_value=1.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            key=f"threshold_{df_key}_{i}"
        )

        # Slider for maximum iterations (meaningful only for MI)
        max_iter = st.slider(
            f"Max Iterations (for MI) ({label})",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key=f"max_iter_{df_key}_{i}"
        )

        # Slider for KNN neighbors (meaningful only for KNN imputation)
        n_neighbors = st.slider(
            f"N Neighbors (for KNN) ({label})",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key=f"n_neighbors_{df_key}_{i}"
        )

        config.append((method, columns, subgroup, threshold, max_iter, n_neighbors))

    return config


def apply_cleaning(df, config):
    """
    Apply all cleaning methods specified in the configuration to the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame to clean.
    - config (list): List of tuples containing method configurations.

    Returns:
    - pd.DataFrame: Cleaned DataFrame after applying all methods sequentially.
    """
    for method, columns, subgroup, threshold, max_iter, n_neighbors in config:
        df = apply_method(df, method, columns, subgroup, threshold, max_iter, n_neighbors)
    return df
