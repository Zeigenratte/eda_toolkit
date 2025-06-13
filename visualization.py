from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set consistent visual styles for matplotlib and seaborn plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")

# Define measurement units for each relevant column in the dataset
column_units = {
    'N_Days': 'days',
    'Status': '',  # Possible values: C, CL, D
    'Drug': '',  # Possible values: D-penicillamine, Placebo, No Drug
    'Age': 'years',
    'Sex': '',  # Possible values: M, F
    'Ascites': '',  # Possible values: N, Y
    'Hepatomegaly': '',  # Possible values: N, Y
    'Spiders': '',  # Possible values: N, Y
    'Edema': '',  # Possible values: N, S, Y
    'Bilirubin': 'mg/dl',
    'Cholesterol': 'mg/dl',
    'Albumin': 'gm/dl',
    'Copper': 'μg/day',
    'Alk_Phos': 'U/liter',
    'SGOT': 'U/ml',
    'Tryglicerides': 'mg/dl',
    'Platelets': 'ml/1000',
    'Prothrombin': 's',
    'Stage': ''  # Possible values: 1, 2, 3, 4
}

class MissingValueSubsetComparator:
    """
    Compare and visualize missing value patterns between a specified subgroup and the complete dataset.

    This class supports:
        - Computing counts of missing values for selected columns in both the entire dataset and a defined subgroup.
        - Producing a grouped bar plot to facilitate visual comparison of missing data prevalence.

    Parameters
    ----------
    df : pd.DataFrame
        The complete dataset.
    subgroup : str
        A pandas query string defining the subgroup, e.g., "Drug == 'No Drug'".
    columns : list, optional
        List of columns to include in the analysis.
        If None, all columns with at least one missing value are considered.
    """

    def __init__(self, df: pd.DataFrame, subgroup: str, columns: list = None):
        self.df = df
        self.subgroup_query = subgroup
        self.columns = (
            columns if columns is not None
            else df.columns[df.isna().any()].tolist()
        )
        self.subgroup = df.query(subgroup)

    def compute_missing_summary(self) -> pd.DataFrame:
        """
        Compute a summary DataFrame with missing value counts for each specified column
        comparing the complete dataset against the subgroup.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ['Group', 'Measure', 'Missing Count'], where 'Group' distinguishes
            between the complete dataset and the subgroup.
        """
        summary = []
        for col in self.columns:
            count_all = self.df[col].isna().sum()
            count_sub = self.subgroup[col].isna().sum()
            summary.append({
                'Group': 'Complete Group',
                'Measure': col,
                'Missing Count': count_all
            })
            summary.append({
                'Group': 'Subgroup',
                'Measure': col,
                'Missing Count': count_sub
            })
        return pd.DataFrame(summary)

    def plot_missing_comparison(self):
        """
        Visualize missing value counts as grouped bar plots for the complete dataset and the subgroup.

        The plot features:
            - Bars grouped by variable/measure.
            - Hue distinguishes between complete dataset and subgroup.
            - Annotated missing counts on bars for clarity.
        """
        summary_df = self.compute_missing_summary()

        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

        sns.set_theme(style='whitegrid')

        sns.barplot(
            data=summary_df,
            x='Measure',
            y='Missing Count',
            hue='Group',
            palette='flare',
            edgecolor='black',
            ax=ax
        )

        ax.set_title(
            f"Missing Value Comparison\nSubgroup: {self.subgroup_query}",
            fontsize=18,
            weight='bold'
        )
        ax.set_xlabel("")
        ax.set_ylabel("Number of Missing Values", fontsize=15)

        # Rotate x-axis labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

        ax.legend(fontsize=13, title_fontsize=14)

        # Annotate bars with missing value counts
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%d',
                label_type='edge',
                padding=2,
                fontsize=12
            )

        # Style axis spines with black edges and thicker lines
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

        # Render the plot in Streamlit and close the figure to free memory
        st.pyplot(fig)
        plt.close(fig)

class DistributionPlot:
    """
    A class for visualizing the distribution of numeric and categorical variables
    across one or more pandas DataFrames, with support for subgroups and column selection.

    Parameters
    ----------
    dataframes : dict of {str: pd.DataFrame}
        Dictionary of DataFrames to compare, e.g., {'Original': df1, 'Transformed': df2}.
        All DataFrames must have the same columns.
    """

    def __init__(self, dataframes: dict):
        self.dataframes = dataframes
        # Extract column names from the first dataframe in the dictionary
        self.column_names = list(next(iter(dataframes.values())).columns)

    def _get_subgroup_mask(self, df: pd.DataFrame, subgroup):
        """
        Generate a boolean mask to select a subgroup of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame on which to apply the subgroup filter.
        subgroup : None, str, or pd.Series
            - None: no filtering (all True).
            - str: pandas query string (e.g., "Drug == 'No Drug'").
            - pd.Series of bool: directly used as mask.

        Returns
        -------
        pd.Series
            Boolean mask indicating rows belonging to the subgroup.

        Raises
        ------
        ValueError
            If `subgroup` is not None, str, or boolean Series.
        """
        if subgroup is None:
            return pd.Series(True, index=df.index)
        elif isinstance(subgroup, str):
            try:
                return df.eval(subgroup)
            except Exception as e:
                raise ValueError(f"Invalid subgroup condition string: '{subgroup}'. Error: {e}")
        elif isinstance(subgroup, pd.Series) and subgroup.dtype == bool:
            if not subgroup.index.equals(df.index):
                raise ValueError("Boolean Series index does not match DataFrame index.")
            return subgroup
        else:
            raise ValueError("Subgroup must be None, a boolean Series, or a valid condition string.")

    def _select_columns(self, df: pd.DataFrame, columns, dtype_include):
        """
        Select columns from the DataFrame based on dtype and optional column list.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to select columns from.
        columns : list or None
            Columns to include. If None, all columns with matching dtype are included.
        dtype_include : list or str
            Data types to include, e.g. ['float64', 'int64'].

        Returns
        -------
        list
            List of columns filtered by dtype and columns parameter.
        """
        if columns is None:
            return df.select_dtypes(include=dtype_include).columns.tolist()
        # Ensure columns are in DataFrame and match dtype
        valid_cols = df.select_dtypes(include=dtype_include).columns
        return [col for col in columns if col in valid_cols]

    def plot_numeric_distributions(self, columns=None, subgroup=None):
        """
        Plot boxplots and histograms/KDEs for numeric columns across dataframes.

        Parameters
        ----------
        columns : dict or None
            Dictionary mapping column names to units, e.g., {'Age': 'years', 'Bilirubin': 'mg/dL'}.
            If None, all numeric columns are included.
        subgroup : None, str, or pd.Series
            Defines subgroup to filter rows, passed to `_get_subgroup_mask`.
        """
        all_data = []
        units = columns if columns is not None else {}

        # Aggregate data from all dataframes with subgroup filtering
        for source_name, df in self.dataframes.items():
            df_filtered = df.copy()
            mask = self._get_subgroup_mask(df_filtered, subgroup)
            df_filtered = df_filtered.loc[mask]
            numeric_cols = self._select_columns(df_filtered, columns, ['float64', 'int64'])
            for col in numeric_cols:
                # Append all non-missing values with source and column info
                for val in df_filtered[col].dropna():
                    all_data.append({
                        'Column': col,
                        'Value': val,
                        'Source': source_name
                    })

        if not all_data:
            st.warning("No data available for plotting numeric distributions with the given filters.")
            return

        plot_df = pd.DataFrame(all_data)
        unique_columns = plot_df['Column'].unique()
        n_vars = len(unique_columns)
        cols = 3
        rows = int(np.ceil(n_vars / cols)) * 2  # Two rows per variable: boxplot + histogram

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(6 * cols, 2.8 * rows),
            constrained_layout=True,
            gridspec_kw={'height_ratios': [0.3 if i % 2 == 0 else 1 for i in range(rows)]}
        )

        # Ensure axes is 2D array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.expand_dims(axes, axis=0)
        elif cols == 1:
            axes = np.expand_dims(axes, axis=1)

        sns.set_theme(style="whitegrid")

        for idx, col in enumerate(unique_columns):
            row_group = (idx // cols) * 2
            col_pos = idx % cols
            subset = plot_df[plot_df['Column'] == col]

            # Plot boxplot (horizontal) on the upper row
            ax_box = axes[row_group, col_pos]
            sns.boxplot(
                data=subset,
                x='Value',
                y='Source',
                ax=ax_box,
                palette="flare",
                orient='h',
                width=0.4,
                fliersize=5,
                boxprops=dict(edgecolor='black', linewidth=1.5, alpha=0.8)
            )
            ax_box.set_title(col, fontsize=16, weight='bold')
            ax_box.set_xlabel("")
            ax_box.set_ylabel("")
            ax_box.set_yticks([])
            ax_box.tick_params(axis='x', labelbottom=False)
            ax_box.grid(True, axis='x', linestyle='--', linewidth=0.8)
            for spine in ax_box.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)

            # Plot histogram with KDE on the lower row
            ax_hist = axes[row_group + 1, col_pos]
            sns.histplot(
                data=subset,
                x='Value',
                hue='Source',
                ax=ax_hist,
                bins=35,
                kde=True,
                element="step",
                palette="flare",
                edgecolor='black',
                alpha=0.5
            )
            unit = units.get(col, None)
            xlabel = f"Value ({unit})" if unit else "Value"
            ax_hist.set_xlabel(xlabel, fontsize=13)
            ax_hist.set_ylabel("Count", fontsize=13)
            ax_hist.grid(True, linestyle='--', linewidth=0.8)

            # Fix legend issues in Streamlit by manual handling
            handles, labels = ax_hist.get_legend_handles_labels()
            if handles and labels:
                legend = ax_hist.legend(
                    handles=handles,
                    labels=labels,
                    loc='upper right',
                    frameon=True,
                    fontsize=9,
                    title=""
                )
                if legend:
                    legend.set_title("")

            for spine in ax_hist.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)

        # Remove unused axes (empty subplots)
        total_plots = n_vars
        for r in range(rows):
            for c in range(cols):
                # Calculate corresponding variable index for this subplot
                var_index = (r // 2) * cols + c
                if var_index >= total_plots:
                    fig.delaxes(axes[r, c])

        st.pyplot(fig)
        plt.close(fig)

    def plot_categorical_distributions(self, columns: dict = None, subgroup=None):
        """
        Plot barplots showing counts and percentages for categorical columns
        across the provided DataFrames, optionally filtered by a subgroup.

        Parameters
        ----------
        columns : dict or None
            Dictionary mapping categorical column names to descriptions or units,
            e.g., {'Sex': '', 'Drug': ''}. If None, all categorical columns are included.
        subgroup : None, str, or pd.Series
            Defines subgroup filter applied to each DataFrame. Accepted formats:
            - None: no filtering
            - str: condition string for DataFrame.eval()
            - pd.Series of bool: boolean mask with index matching DataFrame.
        """
        all_data = []

        # Aggregate categorical count and percentage data from each DataFrame
        for source_name, df in self.dataframes.items():
            df_filtered = df.copy()
            mask = self._get_subgroup_mask(df_filtered, subgroup)
            df_filtered = df_filtered.loc[mask]

            # Select categorical columns (object or category dtype)
            cat_cols = self._select_columns(df_filtered, columns, ['object', 'category'])

            for col in cat_cols:
                counts = df_filtered[col].value_counts(dropna=False)
                percentages = df_filtered[col].value_counts(normalize=True, dropna=False) * 100

                # Store each category's count and percentage per source and column
                for category in counts.index:
                    all_data.append({
                        'Column': col,
                        'Category': str(category),
                        'Count': counts[category],
                        'Percentage': percentages[category],
                        'Source': source_name
                    })

        if not all_data:
            st.warning("No categorical data available for plotting with the given filters.")
            return

        plot_df = pd.DataFrame(all_data)
        unique_columns = plot_df['Column'].unique()
        n_vars = len(unique_columns)
        cols = 3
        rows = int(np.ceil(n_vars / cols)) * 2  # Two rows per variable: percentage and count

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(5 * cols, 2.8 * rows),
            constrained_layout=True,
            gridspec_kw={'height_ratios': [0.5 if i % 2 == 0 else 1 for i in range(rows)]}
        )

        # Ensure axes is 2D numpy array for consistent indexing
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.expand_dims(axes, axis=0)
        elif cols == 1:
            axes = np.expand_dims(axes, axis=1)

        sns.set(style="whitegrid")

        for idx, col in enumerate(unique_columns):
            row_group = (idx // cols) * 2
            col_pos = idx % cols
            subset = plot_df[plot_df['Column'] == col]

            # Percentage barplot on the upper row
            ax_pct = axes[row_group, col_pos]
            sns.barplot(
                data=subset,
                x='Category',
                y='Percentage',
                hue='Source',
                ax=ax_pct,
                edgecolor='black',
                palette="flare",
                alpha=0.9
            )
            ax_pct.set_title(col, fontsize=16, weight='bold')
            ax_pct.set_xlabel("")
            ax_pct.set_ylabel("Percentage [%]", fontsize=13)
            ax_pct.set_xticks([])  # Hide x-axis ticks to avoid overlap if many categories

            # Remove legend to avoid duplication; legend shown in count plot below
            if ax_pct.get_legend():
                ax_pct.get_legend().remove()

            # Annotate bars with percentage values, adjusting text color based on bar height
            for container in ax_pct.containers:
                for bar in container:
                    height = bar.get_height()
                    if not np.isnan(height):
                        if height >= 30:
                            ax_pct.text(
                                bar.get_x() + bar.get_width() / 2,
                                height - 5,
                                f"{height:.0f}",
                                ha='center',
                                va='top',
                                color='white',
                                fontsize=10
                            )
                        else:
                            ax_pct.text(
                                bar.get_x() + bar.get_width() / 2,
                                height + 1,
                                f"{height:.0f}",
                                ha='center',
                                va='bottom',
                                color='black',
                                fontsize=10
                            )

            # Beautify plot spines
            for spine in ax_pct.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)

            # Count barplot on the lower row
            ax_count = axes[row_group + 1, col_pos]
            sns.barplot(
                data=subset,
                x='Category',
                y='Count',
                hue='Source',
                ax=ax_count,
                edgecolor='black',
                palette="flare",
                alpha=0.9
            )
            ax_count.set_xlabel("")
            ax_count.set_ylabel("Count", fontsize=13)

            # Format legend and place at upper right, remove title
            legend = ax_count.get_legend()
            if legend:
                legend.set_title("")
                ax_count.legend(loc='upper right', frameon=True, fontsize=9)

            # Annotate bars with count values, adjusting text color similarly
            for container in ax_count.containers:
                for bar in container:
                    height = bar.get_height()
                    if not np.isnan(height):
                        if height > 106:
                            ax_count.text(
                                bar.get_x() + bar.get_width() / 2,
                                height - 8,
                                f"{int(height)}",
                                ha='center',
                                va='top',
                                color='white',
                                fontsize=10
                            )
                        else:
                            ax_count.text(
                                bar.get_x() + bar.get_width() / 2,
                                height + 1,
                                f"{int(height)}",
                                ha='center',
                                va='bottom',
                                color='black',
                                fontsize=10
                            )

            # Beautify plot spines
            for spine in ax_count.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)

        # Remove any unused axes to keep figure clean
        for r in range(rows):
            for c in range(cols):
                var_idx = (r // 2) * cols + c
                if var_idx >= n_vars:
                    fig.delaxes(axes[r, c])

        st.pyplot(fig)
        plt.close(fig)

def heatmap_grid(
    datasets: Dict[str, pd.DataFrame], 
    figsize: Tuple[int, int] = (22, 6)
) -> None:
    """
    Plot Spearman correlation heatmaps for multiple DataFrames side by side.

    This function computes the Spearman correlation matrix for each DataFrame
    in the input dictionary and displays the correlation heatmaps in a single
    figure arranged horizontally.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of labeled DataFrames to visualize, e.g., {'Original': df1, 'Transformed': df2}.
    
    figsize : tuple[int, int], optional
        Figure size as (width, height) in inches. Default is (22, 6).

    Returns
    -------
    None
        Displays the heatmaps using Streamlit and closes the figure.
    """
    n = len(datasets)

    # Create subplots: one row, n columns
    fig, axes = plt.subplots(1, n, figsize=figsize)

    # Ensure axes is iterable when n=1
    if n == 1:
        axes = [axes]

    # Iterate over each dataset and its corresponding subplot axis
    for ax, (label, dataset) in zip(axes, datasets.items()):
        # Calculate Spearman correlation on numeric columns
        corr_matrix = dataset.select_dtypes(include=[np.number]).corr(method='spearman')

        # Mask to hide upper triangle (to avoid duplicate info)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot heatmap with annotations
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='rocket',
            annot=True,
            annot_kws={'size': 9},
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(f"{label} Correlation", fontsize=16, weight='bold')

    # Display the figure in Streamlit and close to free memory
    st.pyplot(fig)
    plt.close(fig)
