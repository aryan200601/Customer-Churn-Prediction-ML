# Customer-Churn-Prediction-ML

This project develops an end-to-end Machine Learning pipeline to predict customer churn for a telecommunications company. The goal is to accurately identify customers at high risk of leaving and provide actionable insights to inform retention strategies.

## 🎯 Key Findings and Model Performance

| Metric | Result |
| :--- | :--- |
| **Random Forest AUC-ROC** | [Insert Your Score, e.g., **0.84**] |
| **Model Type** | Random Forest Classifier (chosen over Logistic Regression for better performance) |
| **Top 3 Churn Drivers** | 1. **Contract_Month-to-month** (Highest Risk) <br> 2. **Tenure Months** (Low tenure = High Risk) <br> 3. **InternetService_Fiber Optic** |

**Business Insight:** The model confirms that customers on month-to-month contracts and new users (low tenure) are the primary targets for retention efforts.



## ⚙️ Project Pipeline & Technologies

* **Data Cleaning:** Used **Pandas** to handle missing values (specifically in `Total Charges`) and ensure correct data types.
* **Exploratory Data Analysis (EDA):** Employed **Seaborn** and **Matplotlib** to visualize feature distributions and correlations with churn (e.g., tenure KDE plots, contract count plots).
* **Feature Engineering:** Managed numerical scaling and one-hot encoding for categorical features using **scikit-learn's ColumnTransformer** for an efficient, repeatable workflow.
* **Model Training & Evaluation:** Trained two classification models (**Random Forest** and **Logistic Regression**) using **scikit-learn**, focusing evaluation on **AUC-ROC** and **Recall** due to class imbalance.
* **Interpretation:** Used Random Forest's built-in feature importance to derive actionable business insights.

## 📦 Prerequisites

This project requires Python 3.x and the following libraries:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
