# eda_toolkit
Streamlit-Based Exploratory Data Analysis of Cirrhosis Prediction Dataset
---
This project provides an interactive, modular and extensible Streamlit-based EDA 
tool designed for analysis of Cirrhosis Prediction Dataset.
It includes classes for data cleaning, inconsistency resolution, 
statistical summarization, handling missing values and outliers and 
visual inspection, all accessible through an intuitive web interface.

---
## Features
Data Overview and Inspection
- View dataset shape and structure
- Preview head and tail of the dataset
- Combined descriptive statistics for numeric and categorical features
- Detection and summary of: Missing values and Duplicate rows

Data Cleaning Utilities
- Drop unwanted columns
- Convert column values via custom transformation functions
- Fill missing values with specified constants

Inconsistency Resolution
- Automatic detection and removal of exact duplicate rows
- Identification and replacement of negative values in numeric columns
- Detection of numerical columns suitable for categorical conversion


Handling Missing Values
- Detect missing data by column and subgroups
- Fill missing values using basic or advanced methods
- Report changes by log messages for traceability

Handling Outliers
- Detect outliers by column and subgroups using statistical methods
   Replace outliers with NaN
   Report changes by log messages for traceability

Visualization
- Plot one or multiple datasets for comparrison
- Differentiate between numeric and categorical features
- Available plots: Boxplot, Histplot, Barplot, Heatmap

Interactive EDA
- Run modular analysis via Streamlit widgets
- Combine overviews, summaries, corrections and visualizations in one interface
- Real-time feedback with log messages for traceability
---

