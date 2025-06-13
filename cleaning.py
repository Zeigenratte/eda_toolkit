import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from scipy.stats import skew, zscore

class DataFrameCleaner:
    """
    A customizable data cleaner that allows manual preprocessing operations such as:
        - Dropping columns
        - Converting column values using custom functions
        - Filling missing values

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be cleaned.
    """

    def __init__(self, df: pd.DataFrame):
        self.original_df = df
        self.df = df.copy()

    def drop_columns(self, cols: list) -> None:
        """
        Drop specified columns from the DataFrame.

        Parameters
        ----------
        cols : list
            List of column names to be dropped.
        """
        self.df.drop(columns=cols, inplace=True)
        st.write(f"[FIX] Dropped columns: {cols}")

    def convert_column(self, col: str, func, round_decimals: int = None) -> None:
        """
        Apply a function to a column, optionally rounding the result.

        Parameters
        ----------
        col : str
            Column name to be transformed.
        func : callable
            Function to apply to each element.
        round_decimals : int, optional
            If provided, rounds the transformed values to this number of decimals.
        """
        self.df[col] = self.df[col].apply(func)
        if round_decimals is not None:
            self.df[col] = self.df[col].round(round_decimals)
        st.write(f"[FIX] Converted column '{col}' with function")

    def fillna(self, col: str, value) -> None:
        """
        Fill missing values in a column with a specified value.

        Parameters
        ----------
        col : str
            The column to process.
        value : scalar
            The value to replace NaNs with.
        """
        na_count = self.df[col].isna().sum()
        self.df.loc[self.df[col].isna(), col] = value
        st.write(f"[FIX] Filled {na_count} missing values in '{col}' with {value}")

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df


class InconsistencyResolver:
    """
    Detects and resolves common data inconsistencies in a DataFrame, including:
        - Duplicate rows
        - Negative numeric values
        - Data type misclassification

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be processed.
    name : str, optional
        An optional name for the dataset for logging and reporting purposes (default: "Dataset").
    """

    def __init__(self, df: pd.DataFrame, name: str = "Dataset"):
        self.original_df = df
        self.df = df.copy()
        self.name = name

    def detect_duplicates(self) -> int:
        """
        Detect the number of exact duplicate rows in the DataFrame.

        Returns
        -------
        int
            Count of duplicated rows.
        """
        return self.df.duplicated().sum()

    def resolve_duplicates(self) -> None:
        """
        Remove duplicate rows from the DataFrame and report the count removed.
        """
        count = self.detect_duplicates()
        if count > 0:
            self.df.drop_duplicates(inplace=True)
            st.write(f"[FIX] {count} duplicate rows removed.")
        else:
            st.write("[OK] No duplicate rows found.")

    def detect_negative_values(self) -> dict:
        """
        Detect negative values in numeric columns.

        Returns
        -------
        dict
            Dictionary mapping column names to their count of negative entries.
        """
        negatives = {}
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            count = (self.df[col] < 0).sum()
            if count > 0:
                negatives[col] = count
        return negatives

    def resolve_negative_values(self, replace_with=np.nan) -> None:
        """
        Replace negative values in numeric columns with a specified value (default: NaN).

        Parameters
        ----------
        replace_with : scalar, optional
            The value to replace negative entries with.
        """
        negatives = self.detect_negative_values()
        for col, count in negatives.items():
            self.df[col] = self.df[col].apply(lambda x: x if x >= 0 else replace_with)
            st.write(f"[FIX] {count} negative values in '{col}' replaced with {replace_with}.")
        if not negatives:
            st.write("[OK] No negative values detected.")

    def resolve_dtype_errors(self, max_unique_for_categorical: int = 5) -> None:
        """
        Convert numeric columns with few unique values to 'object' type (categorical).

        Parameters
        ----------
        max_unique_for_categorical : int, optional
            The maximum number of unique values allowed to consider a numeric column as categorical.
        """
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if self.df[col].nunique(dropna=False) <= max_unique_for_categorical:
                self.df[col] = self.df[col].astype('object')
                st.write(f"[FIX] '{col}' converted from numeric to 'object' (categorical).")

    def resolve_all(self) -> None:
        """
        Execute all inconsistency detection and resolution methods sequentially:
            - Duplicate removal
            - Negative value replacement
            - Data type correction
        """
        # 1. Duplicates
        dupes = self.detect_duplicates()
        if dupes > 0:
            st.write(f"[INFO] {dupes} duplicate rows detected.")
        self.resolve_duplicates()

        # 2. Negative values
        negatives = self.detect_negative_values()
        if negatives:
            st.write(f"[INFO] Negative values found in: {list(negatives.keys())}")
        self.resolve_negative_values()

        # 3. Data type errors
        self.resolve_dtype_errors()

    def get_resolved_data(self) -> pd.DataFrame:
        return self.df

class ImputationHandler:

    """
    ImputationHandler provides a suite of methods for handling missing data in a pandas DataFrame.
    
    Supported strategies include:
      - Listwise and pairwise deletion of rows with missing values.
      - Single-value imputation using mean, median, or mode, with automatic selection based on data skewness.
      - Random hot deck imputation, sampling observed values within subgroups.
      - Model-based imputation using KNN or IterativeImputer (MICE) for numeric columns.
    
    Features:
      - Subgroup support: All methods can be applied to a specified subgroup (via boolean mask or query string).
      - Verbose reporting: Optionally prints details about the number and type of values imputed or deleted.
      - Designed for flexible, reproducible, and transparent missing data handling in data science workflows.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be imputed or cleaned. A copy is made internally.

    Methods
    ----------
    listwise_deletion(subgroup=None, verbose=False)
    pairwise_deletion(columns, subgroup=None, verbose=False)
    mean_median_imputation(columns=None, subgroup=None, verbose=False)
    mode_imputation(columns=None, subgroup=None, verbose=False)
    random_hot_deck_imputation(columns=None, subgroup=None, verbose=False)
    multiple_imputation(columns=None, subgroup=None, max_iter=10, random_state=0, verbose=False)
    knn_imputation(columns=None, subgroup=None, n_neighbors=5, weights='uniform', verbose=False)
    """
        
    # === Initialization ===
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # === Helper Methods ===
    def _get_subgroup_mask(self, subgroup):
        """Returns a boolean mask for the specified subgroup."""
        if subgroup is None:
            return pd.Series(True, index=self.df.index)
        elif isinstance(subgroup, str):
            try:
                return self.df.eval(subgroup)
            except Exception as e:
                raise ValueError(f"Invalid subgroup condition string: '{subgroup}'. Error: {e}")
        elif isinstance(subgroup, pd.Series) and subgroup.dtype == bool:
            return subgroup
        else:
            raise ValueError("Subgroup must be None, a boolean Series, or a valid condition string.")

    def _report_changes(self, column_or_original, num_imputed_or_imputed, method, detail=""):
        """Prints a summary of imputation or deletion changes."""
        if isinstance(column_or_original, str):  # Simple imputation
            st.write(f"[{method.upper()}] Column: '{column_or_original}' | Imputed: {num_imputed_or_imputed} values {detail}")
        else:  # Model-based methods
            changes = (column_or_original.isnull().sum() - num_imputed_or_imputed.isnull().sum()).to_dict()
            total = sum(changes.values())
            st.write(f"[{method.upper()}] Total values imputed: {total}")
            for col, val in changes.items():
                if val > 0:
                    st.write(f" - Column '{col}': {val} values imputed")

    # Deletion Methods
    def listwise_deletion(self, subgroup=None, verbose: bool = False) -> pd.DataFrame:
        """
        Removes rows with any missing values within the specified subgroup.

        Parameters
        ----------
        subgroup : None, str, or pd.Series (optional)
            - None: Apply to the whole DataFrame.
            - str: A query string to select a subgroup (e.g., "Sex == 'F'").
            - pd.Series: Boolean mask for selecting rows.
        verbose : bool, default False
            If True, prints the number and percentage of rows removed.

        Returns
        -------
        pd.DataFrame
            The DataFrame after listwise deletion.
        """
        mask = self._get_subgroup_mask(subgroup)
        initial_shape = self.df.shape
        to_drop = self.df[mask].index[self.df[mask].isnull().any(axis=1)]
        self.df = self.df.drop(index=to_drop)
        final_shape = self.df.shape
        if verbose:
            removed = initial_shape[0] - final_shape[0]
            st.write(f"[LISTWISE DELETION] Rows removed: {removed} ({removed / initial_shape[0]:.2%})")
        return self.df

    def pairwise_deletion(self, columns: list, subgroup=None, verbose: bool = False) -> pd.DataFrame:
        """
        Removes rows with missing values in specified columns within the subgroup.

        Parameters
        ----------
        columns : list
            List of column names to check for missing values. Rows with missing values in any of these 
            columns will be removed.
        subgroup : None, str, or pd.Series (optional)
            - None: Apply to the whole DataFrame.
            - str: A query string to select a subgroup (e.g., "Sex == 'F'").
            - pd.Series: Boolean mask for selecting rows.
        verbose : bool, default False
            If True, prints the number and percentage of rows removed, and the number of remaining rows.

        Returns
        -------
        pd.DataFrame
            The DataFrame after pairwise deletion.
        """
        if columns is None:
            raise ValueError("A list of target variables must be provided for pairwise deletion.")
        mask = self._get_subgroup_mask(subgroup)
        initial_shape = self.df.shape
        to_drop = self.df[mask].index[self.df.loc[mask, columns].isnull().any(axis=1)]
        self.df = self.df.drop(index=to_drop)
        final_shape = self.df.shape
        if verbose:
            removed = initial_shape[0] - final_shape[0]
            st.write(f"[PAIRWISE DELETION] Rows removed: {removed} ({removed / initial_shape[0]:.2%})")
            st.write(f"Remaining rows: {final_shape[0]} out of {initial_shape[0]}")
        return self.df

    # === Single Value Imputation ===
    def mean_median_imputation(self, columns: list = None, subgroup=None, verbose: bool = False) -> pd.DataFrame:
        """
        Imputes missing values in numeric columns using mean or median, depending on skewness.

        For each numeric column (or specified columns), missing values are imputed as follows:
        - If the absolute skewness of the observed values is less than 0.5, the mean is used.
        - Otherwise, the median is used.
        - The imputation is performed only within the specified subgroup if provided.

        Parameters
        ----------
        columns : list, optional
            List of column names to impute. If None, all numeric columns are used.
        subgroup : None, str, or pd.Series, optional
            - None: Apply to the whole DataFrame.
            - str: A query string to select a subgroup (e.g., "Sex == 'F'").
            - pd.Series: Boolean mask for selecting rows.
        verbose : bool, default False
            If True, prints the number of values imputed and the method used for each column.

        Returns
        -------
        pd.DataFrame
            The DataFrame after mean/median imputation.
        """
        num_df = self.df.select_dtypes(include=[np.number])
        columns = num_df.columns.tolist() if columns is None else [col for col in columns if col in num_df.columns]
        mask = self._get_subgroup_mask(subgroup)
        for col in columns:
            sub_missing = self.df.loc[mask, col].isnull()
            if sub_missing.any():
                valid_values = self.df.loc[mask & ~self.df[col].isnull(), col]
                if valid_values.empty:
                    if verbose:
                        st.write(f"[SKIPPED] Column: '{col}' | No valid values in subgroup.")
                    continue
                before = sub_missing.sum()
                impute_value = valid_values.mean() if abs(skew(valid_values)) < 0.5 else valid_values.median()
                method = "mean" if abs(skew(valid_values)) < 0.5 else "median"
                self.df.loc[mask & self.df[col].isnull(), col] = impute_value
                if verbose:
                    self._report_changes(col, before, method, f"(value = {impute_value})")
        return self.df

    def mode_imputation(self, columns: list = None, subgroup=None, verbose: bool = False) -> pd.DataFrame:
        """
        Imputes missing values in categorical columns using the mode (most frequent value).

        For each categorical column (or specified columns), missing values are imputed as follows:
        - The mode (most frequent value) among observed values in the (optionally filtered) subgroup is used.
        - If there are no valid (non-missing) values in the subgroup, the column is skipped.
        - The imputation is performed only within the specified subgroup if provided.

        Parameters
        ----------
        columns : list, optional
            List of column names to impute. If None, all categorical columns are used.
        subgroup : None, str, or pd.Series, optional
            - None: Apply to the whole DataFrame.
            - str: A query string to select a subgroup (e.g., "Sex == 'F'").
            - pd.Series: Boolean mask for selecting rows.
        verbose : bool, default False
            If True, prints the number of values imputed and the mode value used for each column.

        Returns
        -------
        pd.DataFrame
            The DataFrame after mode imputation.
        """
        cat_df = self.df.select_dtypes(include=["object", "category"])
        columns = cat_df.columns.tolist() if columns is None else [col for col in columns if col in cat_df.columns]
        mask = self._get_subgroup_mask(subgroup)
        for col in columns:
            sub_missing = self.df.loc[mask, col].isnull()
            if sub_missing.any():
                valid_values = self.df.loc[mask & ~self.df[col].isnull(), col]
                if valid_values.empty:
                    if verbose:
                        st.write(f"[SKIPPED] Column: '{col}' | No valid values in subgroup.")
                    continue
                before = sub_missing.sum()
                mode_value = valid_values.mode().iloc[0]
                self.df.loc[mask & self.df[col].isnull(), col] = mode_value
                if verbose:
                    self._report_changes(col, before, "mode", f"(value = {mode_value})")
        return self.df

    def random_hot_deck_imputation(self, columns: list = None, subgroup=None, verbose: bool = False) -> pd.DataFrame:
        """
        Imputes missing values by randomly sampling from observed (non-missing) values in the specified 
        columns and subgroup.

        For each column (or all columns with missing values if columns is None), missing values are imputed as follows:
        - Within the specified subgroup (if provided), missing values are replaced by randomly sampling 
          (with replacement) from the observed values in that column and subgroup.
        - If there are no valid (non-missing) values in the subgroup for a column, that column is skipped.
        - The imputation is performed only within the specified subgroup if provided.

        Parameters
        ----------
        columns : list, optional
            List of column names to impute. If None, all columns with missing values are used.
        subgroup : None, str, or pd.Series, optional
            - None: Apply to the whole DataFrame.
            - str: A query string to select a subgroup (e.g., "Sex == 'F'").
            - pd.Series: Boolean mask for selecting rows.
        verbose : bool, default False
            If True, prints the number of values imputed for each column.

        Returns
        -------
        pd.DataFrame
            The DataFrame after random hot deck imputation.
        """
        columns = [col for col in self.df.columns if self.df[col].isnull().any()] if columns is None else columns
        mask = self._get_subgroup_mask(subgroup)
        for col in columns:
            sub_missing = self.df.loc[mask, col].isnull()
            if sub_missing.any():
                valid_values = self.df.loc[mask & ~self.df[col].isnull(), col]
                if valid_values.empty:
                    if verbose:
                        st.write(f"[SKIPPED] Column: '{col}' | No valid values.")
                    continue
                before = sub_missing.sum()
                fill_values = np.random.choice(valid_values, size=before, replace=True)
                self.df.loc[mask & self.df[col].isnull(), col] = fill_values
                if verbose:
                    self._report_changes(col, before, "hot deck")
        return self.df

    # === Model-Based Imputation ===
    def multiple_imputation(self, columns: list = None, subgroup=None, max_iter: int = 10, random_state: int = 0, verbose: bool = False) -> pd.DataFrame:
        """
        Performs multiple imputation using IterativeImputer for numeric columns.

        This method uses scikit-learn's IterativeImputer (MICE) to impute missing values in numeric columns.
        For each specified column (or all numeric columns if columns is None), missing values are estimated
        by modeling each feature with missing values as a function of other features in a round-robin fashion.

        Parameters
        ----------
        columns : list, optional
            List of column names to impute. If None, all numeric columns are used.
        subgroup : None, str, or pd.Series, optional
            - None: Apply to the whole DataFrame.
            - str: A query string to select a subgroup (e.g., "Sex == 'F'").
            - pd.Series: Boolean mask for selecting rows.
        max_iter : int, default 10
            Maximum number of imputation iterations.
        random_state : int, default 0
            Random seed for reproducibility.
        verbose : bool, default False
            If True, prints the number of values imputed for each column.

        Returns
        -------
        pd.DataFrame
            The DataFrame after multiple imputation.
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        columns = numeric_df.columns.tolist() if columns is None else [col for col in columns if col in numeric_df.columns]
        mask = self._get_subgroup_mask(subgroup)
        original = self.df[columns].copy()
        temp_df = self.df.loc[mask, columns]
        if temp_df.isnull().sum().sum() == 0:
            if verbose:
                st.write("[MULTIPLE IMPUTATION] No missing values in subgroup.")
            return self.df
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        imputed_values = imputer.fit_transform(temp_df)
        self.df.loc[mask, columns] = imputed_values
        if verbose:
            self._report_changes(original, self.df[columns], "multiple imputation")
        return self.df

    def knn_imputation(self, columns: list = None, subgroup=None, n_neighbors: int = 5, weights: str = 'uniform', verbose: bool = False) -> pd.DataFrame:
        """
        Performs KNN imputation for numeric columns using scikit-learn's KNNImputer.

        For each specified column (or all numeric columns if columns is None), missing values are imputed
        by finding the k-nearest neighbors (rows) based on other feature values and averaging (or weighting)
        their values for the missing entry.

        Parameters
        ----------
        columns : list, optional
            List of column names to impute. If None, all numeric columns are used.
        subgroup : None, str, or pd.Series, optional
            - None: Apply to the whole DataFrame.
            - str: A query string to select a subgroup (e.g., "Sex == 'F'").
            - pd.Series: Boolean mask for selecting rows.
        n_neighbors : int, default 5
            Number of neighboring samples to use for imputation.
        weights : str, default 'uniform'
            Weight function used in prediction. Possible values:
            - 'uniform': uniform weights.
            - 'distance': weight points by the inverse of their distance.
        verbose : bool, default False
            If True, prints the number of values imputed for each column.

        Returns
        -------
        pd.DataFrame
            The DataFrame after KNN imputation.
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        columns = numeric_df.columns.tolist() if columns is None else [col for col in columns if col in numeric_df.columns]
        mask = self._get_subgroup_mask(subgroup)
        original = self.df[columns].copy()
        temp_df = self.df.loc[mask, columns]
        if temp_df.isnull().sum().sum() == 0:
            if verbose:
                st.write("[KNN IMPUTATION] No missing values in subgroup.")
            return self.df
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputed_values = imputer.fit_transform(temp_df)
        self.df.loc[mask, columns] = imputed_values
        if verbose:
            self._report_changes(original, self.df[columns], "knn imputation")
        return self.df
    

class OutliersHandler:
    """
    A class for detecting and handling outliers in a pandas DataFrame using various statistical methods.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze for outliers.

    Attributes
    ----------
    df : pd.DataFrame
        A copy of the input DataFrame.
    numeric_cols : list
        List of column names in the DataFrame that have numeric data types.

    Methods
    -------
    iqr_method(columns=None, subgroup=None, threshold=1.5, verbose=False, remove=False)
    z_score_method(columns=None, subgroup=None, threshold=3.0, verbose=False, remove=False)
    modified_z_score_method(columns=None, subgroup=None, threshold=3.5, verbose=False, remove=False)
    log_iqr_method(columns=None, subgroup=None, threshold=1.5, verbose=False, remove=False)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

    def _get_subgroup_mask(self, subgroup):
        """
        Generate a boolean mask for a subgroup of the DataFrame.

        Parameters
        ----------
        subgroup : None, str, or pd.Series
            If None, selects all rows. If str, evaluates the string as a condition.
            If pd.Series, uses it as a boolean mask.

        Returns
        -------
        pd.Series
            Boolean mask for the DataFrame.
        """
        if subgroup is None:
            return pd.Series(True, index=self.df.index)
        elif isinstance(subgroup, str):
            try:
                return self.df.eval(subgroup)
            except Exception as e:
                raise ValueError(f"Invalid subgroup condition string: '{subgroup}'. Error: {e}")
        elif isinstance(subgroup, pd.Series) and subgroup.dtype == bool:
            return subgroup
        else:
            raise ValueError("Subgroup must be None, a boolean Series, or a valid condition string.")

    def _select_columns(self, columns):
        """
        Select numeric columns from the DataFrame.

        Parameters
        ----------
        columns : list or None
            List of columns to select. If None, selects all numeric columns.

        Returns
        -------
        list
            List of selected numeric columns.
        """
        if columns is None:
            return self.numeric_cols
        return [col for col in columns if col in self.numeric_cols]

    def _report_outliers(self, outlier_mask: pd.DataFrame, method: str):
        """
        Print a summary report of detected outliers.

        Parameters
        ----------
        outlier_mask : pd.DataFrame
            Boolean DataFrame indicating outlier positions.
        method : str
            Name of the outlier detection method.
        """
        total_outliers = outlier_mask.sum().sum()
        st.write(f"[{method.upper()}] Total outliers detected: {total_outliers}")
        for col in outlier_mask.columns:
            count = outlier_mask[col].sum()
            if count > 0:
                st.write(f" - Column '{col}': {count} outliers")

    # Statistical Methods
    def iqr_method(self, columns=None, subgroup=None, threshold=1.5, verbose=False, remove=False):
        """
        Detect outliers using the Interquartile Range (IQR) method.

        Parameters
        ----------
        columns : list or None
            Columns to check for outliers. If None, uses all numeric columns.
        subgroup : None, str, or pd.Series
            Subgroup mask or condition string. If None, uses all rows.
        threshold : float
            Multiplier for the IQR to define outlier bounds.
        verbose : bool
            If True, prints a summary of detected outliers.
        remove : bool
            If True, replaces outliers with NaN in the DataFrame.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame indicating outlier positions.
        """
        columns = self._select_columns(columns)
        mask = self._get_subgroup_mask(subgroup)
        outliers = pd.DataFrame(False, index=self.df.index, columns=columns)

        for col in columns:
            data = self.df.loc[mask, col]
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers.loc[mask, col] = (data < lower) | (data > upper)

        if verbose:
            self._report_outliers(outliers.loc[mask], method="IQR")

        if remove:
            self.df.loc[:, columns] = self.df.loc[:, columns].mask(outliers)
            return self.df
        
        return outliers

    def z_score_method(self, columns=None, subgroup=None, threshold=3.0, verbose=False, remove=False):
        """
        Detect outliers using the Z-score method.

        Parameters
        ----------
        columns : list or None
            Columns to check for outliers. If None, uses all numeric columns.
        subgroup : None, str, or pd.Series
            Subgroup mask or condition string. If None, uses all rows.
        threshold : float
            Z-score threshold for outlier detection.
        verbose : bool
            If True, prints a summary of detected outliers.
        remove : bool
            If True, replaces outliers with NaN in the DataFrame.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame indicating outlier positions.
        """
        columns = self._select_columns(columns)
        mask = self._get_subgroup_mask(subgroup)
        outliers = pd.DataFrame(False, index=self.df.index, columns=columns)

        for col in columns:
            data = self.df.loc[mask, col].dropna()
            z_scores = zscore(data)  # Standard Z-score calculation
            outliers.loc[data.index, col] = np.abs(z_scores) > threshold

        if verbose:
            self._report_outliers(outliers.loc[mask], method="Z-Score")

        if remove:
            self.df.loc[:, columns] = self.df.loc[:, columns].mask(outliers)
            return self.df
        
        return outliers

    def modified_z_score_method(self, columns=None, subgroup=None, threshold=3.5, verbose=False, remove=False):
        """
        Detect outliers using the Modified Z-score method (based on median and MAD).

        Parameters
        ----------
        columns : list or None
            Columns to check for outliers. If None, uses all numeric columns.
        subgroup : None, str, or pd.Series
            Subgroup mask or condition string. If None, uses all rows.
        threshold : float
            Modified Z-score threshold for outlier detection.
        verbose : bool
            If True, prints a summary of detected outliers.
        remove : bool
            If True, replaces outliers with NaN in the DataFrame.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame indicating outlier positions.
        """
        columns = self._select_columns(columns)
        mask = self._get_subgroup_mask(subgroup)
        outliers = pd.DataFrame(False, index=self.df.index, columns=columns)

        for col in columns:
            data = self.df.loc[mask, col]
            median = data.median()
            mad = np.median(np.abs(data - median))
            if mad == 0:
                continue  # Avoid division by zero if MAD is zero
            modified_z = 0.6745 * (data - median) / mad  # 0.6745 is a scaling constant
            outliers.loc[mask, col] = np.abs(modified_z) > threshold

        if verbose:
            self._report_outliers(outliers.loc[mask], method="Modified Z-Score")

        if remove:
            self.df.loc[:, columns] = self.df.loc[:, columns].mask(outliers)
            return self.df

        return outliers

    def log_iqr_method(self, columns=None, subgroup=None, threshold=1.5, verbose=False, remove=False):
        """
        Detect outliers using the IQR method on log-transformed data.

        Parameters
        ----------
        columns : list or None
            Columns to check for outliers. If None, uses all numeric columns.
        subgroup : None, str, or pd.Series
            Subgroup mask or condition string. If None, uses all rows.
        threshold : float
            Multiplier for the IQR to define outlier bounds on log-transformed data.
        verbose : bool
            If True, prints a summary of detected outliers.
        remove : bool
            If True, replaces outliers with NaN in the DataFrame.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame indicating outlier positions.
        """
        columns = self._select_columns(columns)
        mask = self._get_subgroup_mask(subgroup)
        outliers = pd.DataFrame(False, index=self.df.index, columns=columns)

        for col in columns:
            # Use np.log(x) if x > 0 else np.nan to avoid log(0) or log(negative)
            log_col = self.df.loc[mask, col].apply(lambda x: np.log(x) if x > 0 else np.nan)
            Q1 = log_col.quantile(0.25)
            Q3 = log_col.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers.loc[mask, col] = (log_col < lower) | (log_col > upper)

        if verbose:
            self._report_outliers(outliers.loc[mask], method="Log-IQR")

        if remove:
            self.df.loc[:, columns] = self.df.loc[:, columns].mask(outliers)
            return self.df

        return outliers