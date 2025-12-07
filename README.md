# Meta Analytics: Real Estate Price Prediction & Contextual Analysis

This project performs a comprehensive analysis of the **"Inca Tribe House Prices"** dataset. It combines traditional Data Science techniques (EDA, Preprocessing, Statistical Analysis) with Large Language Models (LLMs) to infer context and metadata about the dataset automatically.

## 📋 Project Overview

The goal of this project is to analyze real estate data to understand market trends, clean and prepare the data for regression modeling, and utilize Generative AI to provide semantic understanding of the data features.

### Key Objectives
1.  **Exploratory Data Analysis (EDA):** Visualize distributions, correlations, and outliers.
2.  **Meta Analysis via LLM:** Use Hugging Face Transformers (GPT-2) to automatically generate definitions for column headers and infer the geographical/contextual origin of the dataset based on data samples.
3.  **Data Preprocessing:** Implement imputation strategies, scaling, and feature engineering to prepare the dataset for price prediction models.

## 🛠 Tech Stack

*   **Language:** Python
*   **Environment:** Google Colab / Jupyter Notebook
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (Imputation, Scaling)
*   **GenAI / LLM:** Hugging Face `transformers`, `accelerate`, `datasets`

## 📂 Dataset

The project uses the file `Inca Tribe House Prices.csv`.
**Key Features:**
*   `Type` (e.g., Apartment, Villa, Chalet)
*   `Price` (Target Variable)
*   `Area` (Square meters)
*   `Bedrooms` & `Bathrooms`
*   `City` & `Compound`
*   `Delivery_Date` & `Payment_Option`

## 🚀 Installation & Setup

If running locally, ensure you have the required libraries installed:

```bash
pip install pandas matplotlib seaborn scikit-learn transformers datasets accelerate
```

## 📊 Methodology

### 1. Data Cleaning & EDA
*   **Type Conversion:** Ensuring `Price` and `Area` are numeric, handling non-numeric errors.
*   **Visualization:**
    *   Histograms to view Price distribution.
    *   Correlation Heatmaps to identify relationships between variables.
    *   Boxplots to detect outliers in Price relative to Property Type.
    *   Geographic analysis via City counts.

### 2. LLM Integration (Meta Analytics)
We leverage the **GPT-2** model via the Hugging Face API to analyze the raw data structure.
*   **Input:** A string sample of the dataframe head.
*   **Prompt:** *"...Based on this table, where do you think it comes from? What does this data represent?"*
*   **Output:** The model generates a contextual description of the columns, helping to automate the "Data Dictionary" creation process.

### 3. Preprocessing Pipeline
*   **Imputation:** Handling missing values in `Bedrooms`, `Bathrooms`, and `Area` using `SimpleImputer` (Strategies: Mean/Median).
*   **Feature Engineering:**
    *   Created `Total_Rooms` (`Bedrooms` + `Bathrooms`).
    *   Created `Room_to_Area_Ratio` to assess spatial density.
*   **Scaling:** Applied `MinMaxScaler` to normalize features for regression algorithms.

## 📉 Results & Insights

*   **Correlations:** Strong positive correlation observed between `Area`, `Bedrooms`, and `Price`.
*   **Missing Data:** Significant missing data found in specific columns, handled via Median imputation to avoid skewing results from outliers.
*   **LLM Context:** The LLM successfully identified the nature of the data (Real Estate/Property Listings) and inferred potential regions based on city names and currency contexts within the prompts.

## 🔮 Future Work

*   **Model Training:** Train Regression models (Linear Regression, Random Forest, XGBoost) to predict `Price`.
*   **Hyperparameter Tuning:** Optimize model performance.
*   **Deployment:** Create a Streamlit app to allow users to input property details and get a price estimate.

---

**Author:** [Your Name]
**Date:** 2023
