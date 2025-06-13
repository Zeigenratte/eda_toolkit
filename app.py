import streamlit as st
import pandas as pd
import warnings
from sklearn.experimental import enable_iterative_imputer
from overview import OverviewAnalysis
from cleaning import DataFrameCleaner, InconsistencyResolver
from cleaning import ImputationHandler, OutliersHandler
from visualization import MissingValueSubsetComparator, DistributionPlot
from visualization import heatmap_grid, column_units
from interact import apply_method, cleaning_ui

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Biotech Data Explorer", layout="wide")

st.title("Exploratory Data Analysis (EDA) of Biotechnological Data")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Navigation",
    [
        "📂 Load Data",
        "📝 Description",
        "🧬 Data Structure Overview",
        "🧹 Data Cleaning and Resolving Inconsistencies",
        "⭕ Missing Values Exploration",
        "👥 Outliers Exploration",
        "🧪 Example 1",
        "🔬 Example 2",
        "🎛️ Interactive Exploration",
        "📌 Considerations",
        "📄 Summary",
        "📚 Literature"
    ]
)

# Load Data
if menu == "📂 Load Data":
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state.df_raw = df_raw
        st.success("Data loaded successfully!")
        st.write("Preview of Raw Data:")
        st.dataframe(df_raw.head())

# Dataset Description from kaggle
elif menu == "📝 Description":
    st.header("Description - Cirrhosis Prediction Dataset")
    st.markdown("""
    ### Context

    Cirrhosis is a late stage of scarring (fibrosis) of the liver caused
    by many forms of liver diseases and conditions, such as hepatitis and 
    chronic alcoholism.
    The following data contains the information collected from the Mayo 
    Clinic trial in primary biliary cirrhosis (PBC) of the liver conducted 
    between 1974 and 1984. A description of the clinical background for the 
    trial and the covariates recorded here is in: 
    - Chapter 0, especially Section 0.2 of Fleming and Harrington, *Counting 
      Processes and Survival Analysis*, Wiley, 1991.  

    A more extended discussion can be found in:  
    - Dickson, et al., *Hepatology* 10:1–7 (1989).  
    - Markus, et al., *N Eng J of Med* 320:1709–13 (1989).

    A total of 424 PBC patients, referred to Mayo Clinic during that ten-
    year interval, met eligibility criteria for the randomized placebo-
    controlled trial of the drug D-penicillamine.
    The first 312 cases in the dataset participated in the randomized trial 
    and contain largely complete data. The additional 112 cases did not 
    participate in the clinical trial but consented to have basic 
    measurements recorded and to be followed for survival. Six of those 
    cases were lost to follow-up shortly after diagnosis, so the data here 
    are on an additional 106 cases as well as the 312 randomized 
    participants. [1]
 
    ### Attribute Information

    1. **ID:** unique identifier  
    2. **N_Days:** number of days between registration and the earlier of 
    death, transplantation, or study analysis time in July 1986  
    3. **Status:** status of the patient  
        - `C` (censored)  
        - `CL` (censored due to liver transplant) 
        - `D` (death)  
    4. **Drug:** type of drug — `D-penicillamine` or `placebo`  
    5. **Age:** age in [days]  
    6. **Sex:** `M` (male) or `F` (female)  
    7. **Ascites:** presence of ascites — `N` (No) or `Y` (Yes)  
    8. **Hepatomegaly:** presence of hepatomegaly — `N` (No) or `Y` (Yes)  
    9. **Spiders:** presence of spiders — `N` (No) or `Y` (Yes)  
    10. **Edema:**  
        - `N`: no edema and no diuretic therapy for edema  
        - `S`: edema present without diuretics, or edema resolved by 
          diuretics  
        - `Y`: edema despite diuretic therapy  
    11. **Bilirubin:** serum bilirubin in [mg/dl]  
    12. **Cholesterol:** serum cholesterol in [mg/dl]  
    13. **Albumin:** albumin in [gm/dl]  
    14. **Copper:** urine copper in [µg/day]  
    15. **Alk_Phos:** alkaline phosphatase in [U/liter]  
    16. **SGOT:** SGOT in [U/ml]  
    17. **Tryglicerides:** tryglicerides in [mg/dl]  
    18. **Platelets:** platelets per cubic [ml/1000]  
    19. **Prothrombin:** prothrombin time in seconds [s]  
    20. **Stage:** histologic stage of disease (1, 2, 3, or 4)  
    """)

elif menu == "🧬 Data Structure Overview":
    st.header("Overview of Data Structure")
    st.markdown("""
    This section provides an overview of the structure and contents of the 
    **Cirrhosis Prediction Dataset**, including variable types, formats, and 
    how the data is structured for analysis.
    """)

    if "df_raw" in st.session_state:
        df = st.session_state.df_raw

        # Create an overview analysis object for the raw dataset
        overview = OverviewAnalysis(df, name="Cirrhosis Prediction Dataset")

        st.subheader("Data Preview")
        st.markdown("""
        - The dataset consists of **418 observations** with **20 variables** 
          capturing clinical and biochemical characteristics of patients evaluated 
          for cirrhosis. 
          The first column, `ID`, is a unique identifier for each patient and will 
          typically be excluded. The variable `Age` is originally recorded in days 
          rather than years and should be converted to years for easier 
          interpretability.

        - The **first 312 rows** represent patients who participated in a 
          **randomized clinical trial**. These entries contain comprehensive 
          information across all variables, including drug treatment, liver 
          function indicators, and laboratory results.

        - The **remaining 106 rows** correspond to patients who did **not** 
          participate in the **randomized clinical trial** but consented to data 
          collection for survival analysis. These entries exhibit substantial 
          missingness in several clinical and laboratory variables, though basic 
          metrics like `Age`, `Sex`, `Edema`, and `Platelets` are often present.
                    
        [2]
        """)

        # Preview head and tail for the raw dataset
        overview.data_preview()

        st.subheader("Summarize Overview")
        st.markdown("""
        - The dataset contains both **numerical and categorical variables**, 
          with some columns containing incomplete records, especially among the 
          last 106 patients, as seen by reduced count values. High standard 
          deviations (e.g., `Alk_Phos` ≈ 2140) suggest wide variability or 
          potential outliers. 
          The `Stage` variable, currently typed as `float64`, represents an 
          ordinal categorical variable and should be converted to `object` 
          type for further operations.
        """)
        # Preview descriptive stats for numeric and categorical columns for 
        # the raw dataset
        overview.summarize_overview()

        st.subheader("Missing Values")
        st.markdown("""
        - Most variables are complete for the majority of patients, but there is 
          substantial missingness in several clinical and laboratory variables. 
          Notably, the `Drug`, `Ascites`, `Hepatomegaly`, and `Spiders` columns 
          have **106 missing values** each, corresponding to patients who did not 
          participate in the clinical trial. Laboratory variables such as 
          `Cholesterol`, `Copper`, and `Triglycerides` also have a significant 
          proportion of missing entries in the randomized clinical trial. 

        - The overall missing data rate is **12.4%**, indicating that careful 
          handling of missing values will be necessary for robust analysis.
        """)
        # Summary for missing values for the raw dataset
        overview.missing_values()

        st.subheader("Duplicate Summary")
        st.markdown("""
        - The dataset doesn't contain any duplicated rows.
        """)
        # Summary for duplicates in the raw dataset
        overview.duplicates_summary() 

    else:
        st.warning("Please upload a dataset first in the 'Load Data' section.")


elif menu == "🧹 Data Cleaning and Resolving Inconsistencies":
    st.header("Data Cleaning")
    st.markdown("""
    The following steps are specifically performed to clean the dataset 
    based on our observations from the Overview Analysis.

    - The `ID` column is removed, as it does not contribute to the analysis. 
    - `Age` is converted from days to years for better interpretability. 
    - Patients who did not participate in the clinical trial are assigned 
      the label **No Drug** in the `Drug` variable. 
    """)


    if "df_raw" in st.session_state:
        df = st.session_state.df_raw

        # Create a DataFrameCleaner instance for the working DataFrame
        cleaner = DataFrameCleaner(df)

        # Drop identifier column
        cleaner.drop_columns(['ID'])

        # Convert 'Age' from days to years using a lambda function,
        # rounding to the nearest integer
        cleaner.convert_column('Age', lambda x: x / 365, round_decimals=0)

        # Assign "No Drug" to patients who did not participate in the
        # clinical trial
        cleaner.fillna('Drug', 'No Drug')

        # Retrieve the cleaned DataFrame
        df_clean = cleaner.get_cleaned_data()

        st.header("Inconsistencies")
        st.markdown("""
        Before further analysis, common types of inconsistencies in the dataset 
        are addressed:

        - The dataset is checked for exact duplicate rows. No duplicates are 
          found, as expected from the duplicate summary.
        - Numeric columns are examined for negative values, which are not 
          expected in clinical or laboratory measurements. No negative values 
          are found.
        - Some numeric columns, such as `Stage`, represent ordinal information 
          and are converted from `float64` to `object` type for proper handling.
        """)

        # Create an instance of the inconsistency resolver for the current 
        # DataFrame
        resolver = InconsistencyResolver(df_clean)

        # Run all inconsistency detection and resolution steps 
        # (duplicates, negatives, dtypes)
        resolver.resolve_all()

        # Retrieve the consistent DataFrame after resolving inconsistencies
        df_consistent = resolver.get_resolved_data()
        st.session_state.df_consistent = df_consistent

    else:
        st.warning("Please upload a dataset first in the 'Load Data' section.")

elif menu == "⭕ Missing Values Exploration":
    st.header("Missing Values Exploration")
    st.markdown("""
    This section explores the pattern and extent of missing values in the
    cleaned and consistent dataset. Several variables, especially those
    related to clinical and laboratory measurements (`Ascites`,
    `Hepatomegaly`, `Spiders`, `Cholesterol`, `Copper`, `Alk_Phos`,
    `SGOT`, `Tryglicerides`, `Platelets`, `Prothrombin`, and `Stage`) 
    contain missing entries. The missingness is most pronounced among
    patients who did not participate in the clinical trial, as indicated
    by the "No Drug" label.

    Notably, there are different types of missing data.

    - Missing Completely at Random (**MCAR**): Data is missing independently
      of both observed and unobserved values—there is no systematic
      pattern. (Example: A doctor forgets to record gender for every sixth
      patient regardless of their characteristics.)

    - Missing at Random (**MAR**): Missingness depends only on observed
      data, not on the missing values themselves. This allows for valid
      statistical modeling if the observed variables related to the
      missingness are included. (Example: Older patients are less likely 
      to report a history pneumonia, so missingness in pneumonia depends 
      on age.)

    - Missing Not at Random (**MNAR**): The probability of missingness
      depends on both observed and unobserved data, making it difficult
      to handle without assumptions. (Example: Patients with low blood 
      pressure are more likely to have missing blood pressure readings, 
      and the missingness is directly related to the value itself.)
                
    [3]

    General steps to handle missing values are as follows.

    1. Identify patterns and reasons for missing data.

    2. Analyse the proportion of missing data.

    3. Choose the best imputation method.
                
    [3]

    Let's examine the missing data more closely. Given that the 106
    non-randomized patients constitute a substantial portion of the
    overall missing values, we will compare the proportion of missing
    data within the "No Drug" subgroup to that of the entire cohort.   
    """)

    if "df_consistent" in st.session_state:
        df_consistent = st.session_state.df_consistent
        # Create a MissingValueSubsetComparator for all columns with missing data
        mvc = MissingValueSubsetComparator(df_consistent, subgroup="Drug == 'No Drug'")

        # Plot comparison
        mvc.plot_missing_comparison()

        st.markdown("""
        The 106 non-randomized patients show **systematic absence** of clinical 
        symptom and lab variables in `Ascites`, `Hepatomegaly`, `Spiders`, 
        `Cholesterol`, `Copper`, `Alk_Phos`, `SGOT` and `Tryglicerides`.
        This is a case of **MNAR**, as missingness is tied to non-participation 
        in the trial, not to random chance. The "No Drug" group accounts for 
        about a quarter of the dataset, so dropping it entirely isn’t an option. 
        If possible it should remain part of the analysis, with appropriate handling.

        Let's examine the other cases not tied to non-participation in the trial. 
        """)

        # Create a MissingValueSubsetComparator for selected columns in the 'No Drug' subgroup
        mvc_selected = MissingValueSubsetComparator(
            df_consistent, 
            "Drug == 'No Drug'", 
            columns=[
                'Cholesterol', 
                'Copper', 
                'Tryglicerides', 
                'Platelets', 
                'Prothrombin', 
                'Stage'
                ]
        )

        # Plot the comparison of missing values for these columns between the full dataset and 
        # the 'No Drug' subgroup
        mvc_selected.plot_missing_comparison()

        st.markdown("""
        The randomized group has missing values in `Cholesterol`, `Copper`, 
        `Tryglicerides`, and `Platelets`. These appear to be **MAR** or possibly 
        **MCAR**, as the missingness occurs without a structural cause. This 
        makes standard imputation feasible. For the variables `Platelets`, 
        `Prothrombin` and `Stage` a few entries are missing from non-randomized 
        patients. Since the variable `Stage` could be a target variable in 
        supervised modeling, it is a consideration to remove those rows. 

        Overall given the distinct patterns of missingness, it may be useful 
        to split the data into randomized and non-randomized cohorts for further 
        analysis or supervised modeling.
        """)

        st.header("Missing Value Methods")
        st.markdown("""
        After exploring the extent and nature of missingness, the next step is 
        choosing how to deal with it. The right method depends on how much 
        data is missing and why it's missing. Poor handling of missing values 
        can skew statistical results, introduce bias, or reduce the accuracy.

        Most imputation strategies fall into three categories.

        - **Deletion Methods**, where incomplete rows or values are dropped.
        - **Single Value Imputations**, which replace missing values with 
          simple estimates like the mean, median, or mode.
        - **Model-Based Imputation**, which uses other variables to predict 
          missing values.

        Each method has its own assumptions and trade-offs. The following 
        sections summarize some select key techniques, their appropriate 
        use cases, and limitations.                     
        [3]

        #### Deletion Methods
                    
        1. **Complete-Case Analysis (Listwise Deletion)**: Removes rows with 
        any missing values, ensuring that only fully observed cases are used 
        in the analysis.
        - Use when: Missingness is minimal and MCAR.
        - Avoid when: Missingness relates to other variables — can bias 
          results and reduce sample size.

        2. **Available-Case Analysis (Pairwise Deletion)**: Includes all 
        available data for each analysis, excluding only missing variables 
        per case.
        - Use when: You want to keep more data for variable-specific analyses.
        - Avoid when: You need consistent sample sizes across 
          analyses — can lead to biased comparisons.

        #### Single Value Imputation
        1. **Mean and Median Imputation**: Replaces missing numeric values 
        with the column’s mean or median (median is better with outliers).
        - Use when: Data is numeric and missingness is low and likely MCAR.
        - Avoid when: Preserving variance or correlations is important. 
          Not suitable for categorical data.

        2. **Mode Imputation**: Fills in missing categorical values with 
        the most frequent category.
        - Use when: The variable is categorical and has a clear dominant value.
        - Avoid when: The mode is weak or unclear (e.g., uniform distributions).

        3. **Random Hot Deck Imputation**: A missing value is imputed from a 
        randomly selected observed values from similar records within the 
        same dataset.
        - Use when: There are enough complete cases and meaningful groupings 
          to define similarity.
        - Avoid when: Similar records are small, poorly matched, or similarity 
          is unclear.
                    
        [3], [4]

        #### Model-Based Imputation
        1. **Multiple Imputation (MI)**: Generates multiple complete datasets 
        by imputing missing values with variability, performs separate analyses, 
        and combines results to reflect uncertainty.
        - Use when: Missingness is MAR, applicable to both categorical and 
          numerical data, and valid statistical inference is required.
        - Avoid when: Data is MNAR or when simplicity is prioritized over 
          statistical rigor.

        2. **K-Nearest Neighbors (kNN)**: Fills missing values using the 
        average of the k most similar complete observations, based on a 
        chosen distance metric.
        - Use when: Data has sufficient complete cases, with meaningful 
          proximity relations, applicable to both categorical and numerical 
          variables.
        - Avoid when: Dataset is small or sparse or similarity between 
          observations is weak or ambiguous. Sensitive to the choice of k.
                    
        [3], [5]
        """)

    else:
        st.warning("Please complete the 'Data Cleaning' step first.")

elif menu == "👥 Outliers Exploration":
    st.header("Outliers Exploration")
    st.markdown("""
    This section examines the presence and nature of outliers in the cleaned 
    and consistent dataset. Outliers refer to observations that 
    significantly deviate from the majority of the data and may arise 
    from a variety of sources. In the medical context, such deviations can 
    result from equipment malfunction, human error, or legitimate but rare 
    patient-specific conditions.

    Outliers may reflect meaningful physiological extremes, such as elevated 
    liver enzymes or unusual platelet counts, or they may be artifacts of 
    incorrect data collection or handling. It is essential to distinguish 
    between these possibilities before deciding how to handle the anomaly. 

    Unchecked outliers can lead to several analytical problems.

    - Increased error variance and reduced statistical power.
    - Decreased normality, particularly when outliers are non-randomly 
      distributed.
    - Biased estimates in predictive modeling, corrupting the true 
      relationship between exposure and outcome.
                
    [3]

    Before proceeding with detection or correction methods, a basic 
    understanding of the data is necessary for selecting an outlier 
    detection method, which depends on factors such as data type, 
    size, distribution, availability of ground truth and the need 
    for interpretability.
    """)

    if "df_consistent" in st.session_state:
        df_consistent = st.session_state.df_consistent
        st.subheader("Numeric Variables")

        # Create a dictionary with the cleaned dataset for comparison
        dict_clean = {
            'Cleaned Dataset': df_consistent
        }
        
        # Initialize the DistributionPlot class with the cleaned dataset dictionary
        dp = DistributionPlot(dict_clean)

        # Plot distributions of categorical variables for the cleaned dataset
        dp.plot_categorical_distributions()
        st.markdown("""
        For **categorical variables**, the bar plots reveal notable category 
        imbalances and - as discussed before - the presence of missing values 
        within several features of the dataset. A prominent example is the 
        `Sex` variable, where female patients are strongly overrepresented 
        in comparison to males. Such disproportions may reflect sampling bias 
        or can have other reasons 
        (e.g., demographic trends in disease prevalence). 
        While these imbalances are not “outliers” in the classical sense, 
        they represent asymmetries that can bias conclusions drawn from the 
        data. True outliers are generally not defined in categorical data in 
        the same way as in numerical data. 

        To enhance interpretability, particularly in the medical context, 
        it is often more insightful to explore categorical variables in 
        relation to a clinically relevant target variable — for instance, 
        mortality (`Status`), treatment group (`Drug`) or disease stage 
        (`Stage`).
        """)

        # Plot distributions of numeric variables for the cleaned dataset, 
        # using column units
        dp.plot_numeric_distributions(columns=column_units)
        st.markdown("""
        The **numerical variables**, as shown in the histogram and boxplot 
        combinations, exhibit varied distributions and clearly visible 
        outliers. Features like `Bilirubin`, `Alk_Phos`, `Copper`, 
        `Triglycerides`, and `SGOT` show heavily right-skewed distributions 
        with long tails and extreme values far from the central mass. 
        Other features such as `Albumin` and `Platelets` are closer to 
        symmetric but still display potential outliers beyond the whiskers 
        of the boxplots. These deviations emphasize the necessity for 
        methods that can handle both skewness and varying distributional 
        shapes.

        The skewed distributions and extreme values without contextual 
        ground truth suggest a need for **distribution-free** and 
        **robust** outlier detection methods. 
        """)

        st.header("Outlier Methods")
        st.markdown("""
        After identifying the distribution and presence of outliers, the next 
        step is to determine how to handle them. Mishandling outliers can lead 
        to distorted results, inflated variance, and biased inferences. 

        In the following we use unsupervised statistical methods, which 
        systematically flag observations that deviate from the central 
        distribution. However, expert knowledge to define plausible value ranges 
        and differentiate between meaningful physiological extremes or incorrect 
        data should complement these methods, since these methods are 
        data-driven and scalable, but they lack domain sensitivity.

        Each statistical method has its advantages and limitations depending on 
        data characteristics and the context of analysis.

        #### Statistical Methods
        1. **Tukey’s Method (IQR Method)**: Detects univariate outliers using 
        the interquartile range. Values beyond 3 IQRs from the median are 
        considered probable outliers.
        - Use when: Data is univariate and distribution-free.
        - Avoid when: Data is skewed or multivariate.

        2. **Z-Score**: Measures the standard deviation distance from the mean. 
        Values with |Z| > 3 are, as a “rule of thumb,” flagged as outliers.
        - Use when: Data is approximately normal and large in size.
        - Avoid when: Distributions are skewed or sample size is small.

        3. **Modified Z-Score**: Uses the median and median absolute deviation 
        instead of mean and standard deviation, making it more robust to 
        extreme values.
        - Use when: Data has outliers that affect the mean.
        - Avoid when: Normality is strongly violated or sample size is very 
        small.

        4. **Log-Transformed IQR (Log-IQR)**: Applies a log transformation to 
        skewed data before performing IQR-based detection.
        - Use when: Data is right-skewed and non-negative.
        - Avoid when: Data contains zero or negative values.
                    
        [3]
        """)

    else:
        st.warning("Please complete the 'Data Cleaning' step first.")

elif menu == "🧪 Example 1":
    st.header("Example 1: Model-Based Imputation with Selective Outlier Deletion")
    st.markdown("""
    ### Missing Values            

    **Example 1**: 
    - The few rows with missing values 
      (`Platelets`, `Prothrombin`, `Stage`) are removed within the the 
      non-randomized participants (`Drug == "No Drug"`). Missing values 
      in the randomized trial subgroup (`Drug != "No Drug"`) for missing 
      clinical variables 
      (`Cholesterol`, `Copper`, `Tryglicerides`, `Platelets`) are imputed 
      using **multiple imputation** to preserve statistical relationships. 
    
    This approach emphasizes data integrity in the non-randomized subgroup 
    by applying **deletion** rather than **imputation**, thereby avoiding 
    assumptions about the non-random missingness. In contrast, 
    **model-based multiple imputation** is applied to the randomized trial 
    participants, yielding a fully imputed dataset for this group. 
    The resulting dataset (`imputed_df`) thus preserves the distinct nature 
    of missingness across subgroups by carrying over unresolved missing 
    values for the non-randomized patients while ensuring completeness 
    where justified.
    """)
    if "df_consistent" in st.session_state:
        df_consistent = st.session_state.df_consistent
        st.write('---')

        # Initialize imputer with the cleaned original DataFrame
        imputer = ImputationHandler(df_consistent)

        # Delete rows with missing values in selected columns for the 
        # non-randomized subgroup
        imputer.pairwise_deletion(
            columns=['Platelets', 'Prothrombin', 'Stage'], 
            subgroup='Drug == "No Drug"', 
            verbose=True
            )
        
        # Apply multiple imputation to specified columns for the 
        # randomized subgroup
        imputer.multiple_imputation(
            columns=['Cholesterol', 'Copper', 'Tryglicerides', 'Platelets'],
            subgroup='Drug != "No Drug"',
            max_iter=10,
            verbose=True
        )

        # Store the resulting DataFrame with imputed values
        imputed_df = imputer.df
        st.session_state.imputed_df = imputed_df
        st.write('---')

        st.subheader("Outliers")
        st.markdown("""   
        **Cleaned NaN DF** (Based on `imputed_df`):
        After applying model-based imputation solely to the randomized 
        subgroup, outliers are identified using a combination of the 
        **modified Z-score** and **log-transformed IQR method**, 
        targeting clinical markers. These outliers are then replaced with 
        "NaN", maintaining a neutral treatment of outliers in the dataset 
        (`outliers_df`) rather than forcing correction.
        """)
        st.write('---')

        # Initialize the outlier handler with the imputed DataFrame
        handler = OutliersHandler(imputed_df)

        # Detect and remove outliers using the modified Z-score method
        handler.modified_z_score_method(
            columns=['Albumin', 'Platelets', 'Prothrombin'], 
            threshold=3.5, 
            verbose=True, 
            remove=True
            )
        
        # Detect and remove outliers using the log-transformed IQR method
        handler.log_iqr_method(
            columns=[
                'SGOT', 
                'Tryglicerides', 
                'Bilirubin', 
                'Copper', 
                'Cholesterol', 
                'Alk_Phos'
                ], 
            threshold=1.5, 
            verbose=True, 
            remove=True
            )
        
        # Store the DataFrame with outliers marked as NaN
        outliers_df = handler.df
        st.session_state.outliers_df = outliers_df
        st.write('---')

        st.markdown("""
        **Cleaned Listwise DF** (Based on `imputed_df`):        
        The variation `outliers_df_2` is a trimmed subset of 
        `outliers_df`. Listwise deletion is applied to rows in the randomized 
        trial subgroup (`Drug != "No Drug"`) that now contain missing values 
        due to outlier removal. Strict exclusion of outliers in trial 
        participants and non-randomized patients ("No Drug") is applied.
        """)
        st.write('---')

        # Initialize imputer with the DataFrame containing NaN-outliers
        imputer = ImputationHandler(outliers_df)

        # Apply listwise deletion to randomized subgroup with remaining 
        # missing values
        imputer.listwise_deletion(subgroup='Drug != "No Drug"', verbose=True)

        # Store the final cleaned dataset after listwise deletion
        outliers_df_2 = imputer.df
        st.write('---')
        
        st.subheader("Visualizastion and Analysis")
        st.markdown("""
        - **Cleaned NaN DF**: After imputation strategies and outlier handlings 
          were applied. Outliers replaced with "NaN".
                    
        - **Cleaned Listwise DF**: After imputation strategies and outlier 
          handlings were applied. Listwise deletion applied to rows with outliers.
                    
        - **Inital DF**: The initial dataset before imputation or outlier 
          handling.
        """)

        # Collect all stages of the data cleaning process for visual comparison
        dict_clean = {
            'Cleaned NaN DF': outliers_df,
            'Cleaned Listwise DF': outliers_df_2,
            'Inital DF': df_consistent
        }

        # Visualize numerical distributions across all DataFrames
        dp = DistributionPlot(dict_clean)
        dp.plot_numeric_distributions(columns=column_units)

        st.markdown("""
        Key observations by examining central tendency, spread, and outlier 
        presence from these distribution plots are as followed. 

        - **Preservation of Shape**: For most variables, the overall shape of 
          the distribution stays pretty consistent across the three versions of 
          the dataset. This suggests that the different imputation strategies 
          succeded at preserving the original statistical structure.

        - **Impact of Listwise Deletion**: Cleaned Listwise DF seems more 
          compressed compared to the others, especially at the edges of the 
          distributions. This shows in `Bilirubin`, `Copper` and `Cholesterol`, 
          where the extreme values are visibly reduced. Removing outliers 
          through listwise deletion also seems to affect surrounding values, 
          suggesting these rows may contain patients with generally abnormal 
          clinical profiles, not just isolated extreme values. This implies 
          that such rows might represent more complex or severe cases, where 
          multiple variables deviate simultaneously.

        - **Effects of Imputation**: Cleaned DF tends to smooth out some of 
          the gaps present in the inital data. Subtle new peaks in 
          `Tryglicerides` and `Copper`, probably caused by mean/median 
          imputation or MI filling in with typical values, which slightly 
          boosts the frequency around central tendencies.

        - **Right-Skewed Variables Still Show Outliers**: Some variables, 
          like `Alk_Phos`, `Cholesterol` and `Copper`, remain very skewed. 
          That’s a sign of real clinical variability rather than just noise.

        The plots validate that preprocessing by combining subgroup-aware 
        imputation with selective outlier removal manages to preserve core 
        distributions while mitigating the influence of missingness and extremes.
        """)

        # Visualize categorical distributions across all DataFrames
        dp.plot_categorical_distributions(columns={'Stage':'', 'Status':'', 'Drug':''})

        st.markdown("""
        However **49 out of 55** removed outliers originate from the randomized 
        trial subgroup, and that most of these patients are classified as 
        **Stage 3 or 4**, holds noteworthy implications for both data analysis 
        and the interpretation of trial outcomes. The fact that most of these 
        outliers are in Stage 3 or 4, which corresponds to more advanced stages 
        of liver disease, emphasizes the clinical plausibility of the values. 
        Rather than random errors, these extreme observations might reflect 
        physiological extremes in patients with severe disease progression. 
        This raises the question of whether these data points should be 
        considered **informative outliers** rather than noise. 

        Automatic removal or imputation of outliers, particularly in such 
        advanced-stage patients, risks eliminating critical information about 
        disease severity, treatment response or patient prognosis. The choice 
        of whether to remove, impute or retain such values should therefore be 
        alligned with expert knowledge and the study’s analytical goals. Also 
        removing outliers concentrated in severe-stage patients will most likely 
        bias the dataset toward milder cases, thereby underestimating variance 
        and potentially skewing treatment effect estimates. The data suggest, 
        that advanced-stage patients should potentially be combined into 
        clinically meaningful subgroups or modeled separately to preserve the 
        heterogeneity of disease progression.

        To summarize, outliers in this dataset are not merely technical 
        anomalies, they likely reflect real clinical extremes concentrated 
        among the randomized and more severely ill patients. Their presence 
        and removal patterns could provide insight into the underlying 
        disease dynamics and imply the need for context-sensitive data 
        handling. These findings underscore the dual role of outliers as 
        potential distorters of analysis when left unchecked, but also as 
        clinically meaningful signals when properly contextualized.
        """)

    else:
        st.warning("Please complete the 'Data Cleaning' step first.")

elif menu == "🔬 Example 2":
    st.header("Example 2: Mean/Mode/Median Imputation for Subgroups")
    st.markdown("""
    ### Missing Values            

    Two imputed DataFrames are created from the same cleaned dataset 
    `df_consistent` for Example 2, each following a structured 
    imputation strategy with a distinct approach for handling the 
    "No Drug" subgroup. (Note: `outliers_df` is processed analog to Example 1, 
    therefore status changes won't be displayed separately.)

    - First, the few rows with missing values 
      (`Platelets`, `Prothrombin`, `Stage`) are removed within the 
      non-randomized participants (`Drug == "No Drug"`). Then missing 
      values in the **randomized trial subgroup** 
      (`Drug != "No Drug"`) for missing clinical variables 
      (`Cholesterol`, `Copper`, `Tryglicerides`, `Platelets`) are 
      imputed using **multiple imputation**. The resulting dataset 
      (`imputed_df`) thus preserves the distinct nature of missingness 
      across subgroups by carrying over unresolved missing values for 
      the non-randomized patients while ensuring completeness where 
      justified.
    
    - For the second imputed DataFrame, the remaining missing variables 
      across the earlier dataset (`imputed_df`)—after imputation in the 
      randomized trial only the non-randomized participants contain missing 
      values—are imputed using **mean or median** and **mode imputation**. 
      This approach balances **model-based imputation** for trial participants 
      with **statistical imputation** for the rest, yielding a complete 
      dataset (`imputed_df_2`) while preserving as much information as 
      possible, albeit disregarding the nature of missingness across subgroups.
    """)

    if "imputed_df" in st.session_state:
        df_consistent = st.session_state.df_consistent
        st.write('---')

        # Instantiate imputer object with consistent dataset
        imputer = ImputationHandler(df_consistent)

        # Apply pairwise deletion for selected features for 'No Drug' subgroup
        imputer.pairwise_deletion(
            columns=['Platelets', 'Prothrombin', 'Stage'], 
            subgroup='Drug == "No Drug"', 
            verbose=True
        )

        # Apply Multiple Imputation (MI) for 'Drug != "No Drug"' subgroup
        imputer.multiple_imputation(
            columns=['Cholesterol', 'Copper', 'Tryglicerides', 'Platelets'],
            subgroup='Drug != "No Drug"',
            max_iter=10,
            verbose=True
        )
        st.write('---')

        # Apply Mean/Median imputation for remaining numeric features
        imputer.mean_median_imputation(verbose=True)

        # Apply Mode imputation for remaining categorical features
        imputer.mode_imputation(verbose=True)
        st.write('---')

        # Store final imputed dataset after all strategies
        imputed_df_3 = imputer.df

        st.subheader("Outliers")
        st.markdown("""
        For both datasets after applying model-based imputation solely to the 
        randomized subgroup, outliers are identified using a combination of the 
        **modified Z-score** and **log-transformed IQR method**, targeting 
        clinical markers. These outliers are then replaced with "NaN", 
        maintaining a neutral treatment of outliers in the datasets rather than 
        forcing correction.
        """)
        st.write('---')

        handler = OutliersHandler(imputed_df_3)

        # Apply Modified Z-Score Method
        handler.modified_z_score_method(
            columns=['Albumin', 'Platelets', 'Prothrombin'], 
            threshold=3.5, 
            verbose=True, 
            remove=True
        )

        # Apply Log-IQR Method
        handler.log_iqr_method(
            columns=[
                'SGOT', 'Tryglicerides', 'Bilirubin', 
                'Copper', 'Cholesterol', 'Alk_Phos'
            ], 
            threshold=1.5, 
            verbose=True, 
            remove=True
        )
        st.write('---')

        # Store outlier-handled version
        outliers_df_3 = handler.df

        st.subheader("Visualization and Analysis")
        st.markdown("""
        - **Partial Imputed DF**: After selective imputation strategies and 
          outlier handling were applied. 
                    
        - **Complete Imputed DF**: After imputation strategies and outlier 
          handlings were applied. Mean/Mode/Median imputation applied for entire
          "No Drug" subgroup.
                    
        - **Initial DF**: The initial dataset before imputation or outlier 
          handling.
        """)

        outliers_df = st.session_state.outliers_df
        df_consistent = st.session_state.df_consistent

        dict_clean = {
            'Partial Imputed DF': outliers_df,
            'Complete Imputed DF': outliers_df_3,
            'Inital DF': df_consistent
        }

        dp = DistributionPlot(dict_clean)
        dp.plot_numeric_distributions(columns=column_units)
        dp.plot_categorical_distributions()

        st.markdown("""
        In **Example 2**, we see how using mean, median, or mode imputation for 
        the "No Drug" subgroup affects the overall shape of the data.

        Most variables in the partially imputed df (`outliers_df`) keep a 
        similar form as the initial dataset (`df_consistent`), meaning the 
        imputation didn’t distort the distributions. However, in the fully 
        imputed version (`outliers_df_3`), variables like `Cholesterol`, 
        `Copper`, `SGOT`, and `Tryglicerides` show sharp, artificial spikes, 
        typical signs of many missing values being filled in with the same 
        average value. The same trend can be observed for categoric variables 
        like `Ascites`, `Hepatomegaly`, or `Spiders`, where imputation with the 
        mode leads to inflated counts for the most common category.

        This kind of imputation ensures dataset completeness but oversimplifies 
        the data, particularly in features with strong skew or where missingness 
        might reflect more than just random gaps.
        """)
    else:
        st.warning("Please complete the 'Example 1' step first.")


elif menu == "🎛️ Interactive Exploration":
    st.subheader("Interactive Exploration of Cleaning Methods")

    if "df_consistent" in st.session_state:
        # Copy the consistent DataFrame for interactive cleaning
        df_original = st.session_state.df_consistent.copy()

        st.markdown("### Select Cleaning Steps for Two DataFrames")

        # UI for configuring cleaning steps for DataFrame A and B
        config_a = cleaning_ui("A", "a")
        config_b = cleaning_ui("B", "b")

        # Button to apply selected cleaning methods to both DataFrames
        if st.button("Apply Cleaning Methods"):
            df_a = df_original.copy()
            for method, columns, subgroup, threshold, max_iter, n_neighbors in config_a:
                df_a = apply_method(df_a, method, columns, subgroup, threshold, max_iter, n_neighbors)

            df_b = df_original.copy()
            for method, columns, subgroup, threshold, max_iter, n_neighbors in config_b:
                df_b = apply_method(df_b, method, columns, subgroup, threshold, max_iter, n_neighbors)

            st.session_state["cleaned_df_a"] = df_a
            st.session_state["cleaned_df_b"] = df_b

            st.success("Cleaning methods applied successfully.")

            overview_a = OverviewAnalysis(df_a, "Cleaned DataFrame A")
            overview_b = OverviewAnalysis(df_b, "Cleaned DataFrame B")
            overview_a.report()
            overview_b.report()
                
        # Button to generate and display distribution plots for both DataFrames
        if st.button("Generate Distribution Plots"):
            df_a = df_original.copy()
            for method, columns, subgroup, threshold, max_iter, n_neighbors in config_a:
                df_a = apply_method(df_a, method, columns, subgroup, threshold, max_iter, n_neighbors)

            df_b = df_original.copy()
            for method, columns, subgroup, threshold, max_iter, n_neighbors in config_b:
                df_b = apply_method(df_b, method, columns, subgroup, threshold, max_iter, n_neighbors)

            st.session_state["cleaned_df_a"] = df_a
            st.session_state["cleaned_df_b"] = df_b

            st.success("Distribution Plots generated successfully.")
            st.subheader("Distribution Plots")

            dict_clean = {
                'DF A': df_a,
                'DF B': df_b,
                'Inital DF': df_original
                }
            
            dp = DistributionPlot(dict_clean)
            dp.plot_numeric_distributions(columns=column_units)
            dp.plot_categorical_distributions()

        # Button to generate and display correlation heatmaps for both DataFrames
        if st.button("Generate Heatmap"):
            df_a = df_original.copy()
            for method, columns, subgroup, threshold, max_iter, n_neighbors in config_a:
                df_a = apply_method(df_a, method, columns, subgroup, threshold, max_iter, n_neighbors)

            df_b = df_original.copy()
            for method, columns, subgroup, threshold, max_iter, n_neighbors in config_b:
                df_b = apply_method(df_b, method, columns, subgroup, threshold, max_iter, n_neighbors)

            st.session_state["cleaned_df_a"] = df_a
            st.session_state["cleaned_df_b"] = df_b

            st.success("Heatmap generated successfully.")
            
            st.subheader("Heatmap")
            st.markdown("""
            **Correlation analysis** examines the relationships between numerical 
            variables in the dataset. By visualizing the **correlation matrix**, 
            we can identify which clinical and laboratory features are strongly 
            associated, either positively or negatively. This helps reveal potential 
            redundancies or key predictors for modeling. 
            """)

            dict_clean = {
                'DF A': df_a,
                'DF B': df_b,
                'Inital DF': df_original
                }
            
            heatmap_grid(dict_clean)

    else:
        st.warning("Please complete the 'Data Cleaning' step first.")

elif menu == "📌 Considerations":
    st.header("Considerations")
    st.markdown("""
    The applied cleaning methods, as previously discussed, are likely to 
    introduce bias into subsequent analyses. Excessive deletion, 
    particularly affecting advanced-stage patients, where outliers are 
    disproportionately removed, can skew the dataset toward milder cases, 
    reducing variance and potentially distort future effect estimates. 
    Also, aggressive imputation, especially using mean, median or mode, 
    likely obscure clinically meaningful variability by smoothing over 
    important deviations, thus limiting the ability to detect true patterns 
    or subgroup differences.

    Given the observed patterns, potential modeling approaches, based on 
    **non-invasive** biomarkers and indicators, or research questions could 
    include the following.

    - Predicting the disease stage variable (`Stage`).
    - Estimating the survival time of patients (`N_Days`, `Status`). [2]
    - Identifying further subgroups with distinct progression profiles.
    """)

elif menu == "📄 Summary":
    st.header("Project Summary")
    st.markdown("""
    This project aimed to perform an **exploratory data analysis (EDA)** on 
    biotechnological data, focusing on improving data quality through 
    structured **cleaning**, **imputation**, and **outlier detection**.     

    ### 1. Data Acquisition and Structure

    A class-based interface was used to download, read, and structure the 
    raw data. Early inspection revealed issues typical of biotechnological 
    datasets: **missing values**, **extreme outliers**, and **heterogeneous
    subgroup characteristics**, especially between randomized trial 
    participants and non-treated patients.

    ### 2. Handling Missing Data

    Two main imputation strategies were employed.

    - **Example 1** emphasized subgroup-sensitive imputation.
      For the **randomized group**, missing values in key clinical variables 
      were filled using **model-based multiple imputation (MI)**, preserving 
      inter-variable relationships. For the **non-randomized group**, rows 
      with missing values were simply **removed** to avoid assumptions about 
      the missingness mechanism.

    - **Example 2** used a more uniform strategy. After imputing the 
      randomized subgroup with MI, the **non-randomized group** was imputed 
      using **mean, median or mode** imputation. This ensured a fully 
      complete dataset, but introduced artificial patterns, especially where 
      missingness was non-random.

    This comparison revealed the **trade-off between data completeness and 
    the preservation of subgroup-specific variance**. While Example 2 
    yielded a structurally complete dataset, it also **blurred distinctions** 
    between clinical subgroups, potentially masking meaningful variability.

    ### 3. Outlier Detection and Handling

    Outliers were detected using two complementary methods.

    - **Modified Z-score** for symmetric distributions,
    - **Log-transformed IQR** for skewed clinical measures.

    Again, two strategies were compared.

    - **Selective deletion** of outliers.
    - **Listwise deletion** of outlier rows.

    Visual analysis showed that **outlier removal compressed the tails** 
    of many distributions, particularly for variables like `Cholesterol`, 
    `Copper`, and `SGOT`. Interestingly, most outliers were concentrated 
    in **advanced-stage patients**, suggesting that **removing these values 
    may bias the dataset toward milder cases** and reduce clinical 
    sensitivity. This underlines the importance of aligning cleaning 
    strategies with **domain expertise** and study goals, rather than relying 
    solely on statistical thresholds.

    ### 4. Data Visualization and Interpretation

    Using libraries such as `Seaborn`, the raw and fully imputed datasets 
    were compared visually. The majority of variable distributions retained 
    their shapes under partial imputation, indicating a minimal impact on 
    global data structure. However, fully imputed versions showed visible 
    artifacts — sharp peaks where too many values were filled with the same 
    average. Categorical features like `Spiders` and `Hepatomegaly` also 
    lost natural variation in the fully cleaned data.

    ### 5. Automation and Dashboard Integration

    All preprocessing steps were implemented in reusable Python classes and 
    integrated into a Streamlit dashboard, enabling users to interactively 
    explore data cleaning decisions and observe their effects on the dataset.
                
    ---
    """)
    st.header("Conclusion")
    st.markdown("""
    This project demonstrates how EDA can be systematically applied to 
    biotechnological data to uncover hidden patterns, assess data quality 
    and guide informed decisions about data cleaning. It highlights the 
    **importance of choosing appropriate strategies based on the nature of 
    missingness and data distribution** and stresses caution when using 
    global imputation or aggressive outlier removal.
    """)

elif menu == "📚 Literature":
    st.header("References")
    st.markdown("""
    1. Cirrhosis Prediction Dataset. (Accessed on 11.06.25, online 
    available under:
    https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset).
                
    2. Dickson, R. E., Grambsch, P. M., Fleming, T. R., Fisher, L. D., 
    & Langworthy, A. (1989). Prognosis in primary biliary cirrhosis: 
    Model for decision making. Hepatology, 10(1), 1–7. 
    https://doi.org/10.1002/hep.1840100102.
                
    3. MIT Critical Data. (2016). Secondary Analysis of Electronic Health 
    Records. Springer International Publishing. 
    https://doi.org/10.1007/978-3-319-43742-2.
              
    4. When to use Mean/Median/Mode imputation. (Accessed on 30.05.25, 
    online available under:
    https://vtiya.medium.com/when-to-use-mean-median-mode-imputation-b0fd6be247db).
                
    5. A Guide to Hot Deck Imputation: Theory, Practice, and Examples. 
    (Accessed on 30.05.25, online available under:
    https://www.numberanalytics.com/blog/guide-hot-deck-imputation).
    """)
