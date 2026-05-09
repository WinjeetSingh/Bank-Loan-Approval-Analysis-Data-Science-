# 🏦 Bank Loan Approval Analysis & Risk Assessment

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Modern_Syntax-150458.svg)
![Seaborn](https://img.shields.io/badge/Data_Viz-Seaborn-4CBB17.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
In the financial sector, assessing the risk of a loan applicant is critical for minimizing default rates and maximizing profitability. This project analyzes historical bank loan data to identify key patterns in applicant demographics, income, and borrowing habits. 

The primary focus of this notebook is to perform rigorous data cleaning, advanced feature engineering, and exploratory data analysis (EDA) to establish a robust foundation for automated risk assessment and future predictive machine learning models.

## 🎯 Key Objectives
1. **Data Imputation & Integrity:** Handle missing data dynamically by applying statistical measures (Mode for categorical/string variables, Median for numerical variables) using modern, warning-free Pandas syntax.
2. **Feature Engineering & Normalization:** Combine financial metrics to create unified features (e.g., `Total_Income`) and apply log transformations to correct extreme right-skewness in financial distributions.
3. **Statistical Correlation:** Evaluate linear relationships between income dimensions and loan amounts using Pearson correlation coefficients.
4. **Heuristic Risk Segmentation:** Develop a custom, logic-based risk metric to classify applicants into actionable business categories (`High Risk` vs. `Low Risk`).

## 🛠️ Methodology & Workflow

### 1. Data Cleaning
Real-world financial data is prone to missing values. The dataset was systematically cleaned:
* Extracted categorical vs. numerical data types dynamically.
* Replaced missing categorical values with the mode to preserve distributional shape.
* Replaced missing numerical values with the median to avoid the undue influence of extreme outliers.
* Utilized modern Pandas `df[col] = ...` reassignment to comply with the latest Copy-on-Write memory management best practices (avoiding `inplace=True` chained assignment warnings).

### 2. Feature Engineering
* **`Total_Income` Creation:** Aggregated `ApplicantIncome` and `CoapplicantIncome` to capture the true purchasing/repayment power of the household.
* **Logarithmic Transformations:** Financial data inherently suffers from exponential skewness. Applied `np.log1p()` to `Total_Income` and `LoanAmount` to compress the long tail, resulting in a pseudo-normal distribution optimized for future algorithmic processing.

### 3. Exploratory Data Analysis (EDA)
* Designed highly readable, customized visualizations using the Seaborn library.
* Analyzed the distribution of requested loan amounts and segmented risk levels.
* Configured plots to adhere to the latest Seaborn API guidelines (e.g., explicit `hue` assignment).

### 4. Risk Assessment
Created a synthesized `Risk_Level` feature based on strict business logic:
* **Low Risk:** Applicants possessing a documented credit history (`Credit_History == 1`) AND a `Total_Income` exceeding $5,000.
* **High Risk:** All other applicants who fail to meet the baseline financial and historical credit thresholds.

## 📊 Key Insights
* **Income Skewness:** The raw income data is highly concentrated on the lower end with significant outliers. Log transformation is mandatory before feeding this data into distance-based machine learning models (like KNN or SVM).
* **Risk Distribution:** By applying the custom heuristic, the dataset can be successfully partitioned, allowing loan officers to immediately filter and prioritize 'Low Risk' applications while flagging 'High Risk' profiles for manual review.

## 💻 Technologies Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Statistical Analysis:** SciPy

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/WinjeetSingh/bank-loan-approval-analysis.git](https://github.com/WinjeetSingh/bank-loan-approval-analysis.git)
