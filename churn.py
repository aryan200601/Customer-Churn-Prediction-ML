import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix


df = pd.read_excel('your_churn_data.xlsx') # Replace with your file path
print(df.head())
print(df.info())



# Convert 'TotalCharges' to numeric, coercing errors (empty strings to NaN)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
# Fill NaN values (which came from the empty strings) with 0 or the mean. 
# Since it's usually for new customers (low 'tenure'), filling with 0 is often appropriate.
df['Total Charges'].fillna(0, inplace=True)



sns.countplot(x='Churn Value', data=df)
plt.title('Churn Distribution')
plt.show() # 
# You'll likely see a significant imbalance (more 'No' than 'Yes'). Mention this in your notes!
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn Value', data=df)
plt.title('Churn Rate by Contract Type')
plt.show() # 
# Insight: Customers with 'Month-to-month' contracts are far more likely to churn.



# Assuming your DataFrame is named 'df'
# Fix: Use 'Tenure Months' and 'Churn Value'
# Plotting Tenure Distribution for non-churned customers (Churn Value == 0)
sns.kdeplot(df.loc[df['Churn Value'] == 0, 'Tenure Months'], label='No Churn', fill=True)

# Plotting Tenure Distribution for churned customers (Churn Value == 1)
sns.kdeplot(df.loc[df['Churn Value'] == 1, 'Tenure Months'], label='Churn', fill=True)

plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Tenure (Months)')
plt.legend()
plt.show()
# Insight: New customers (low tenure) are most likely to churn.
columns_to_drop = [
    'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 
    'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 'CLTV', 
    'Churn Reason'
]


X = df.drop(columns=columns_to_drop + ['Churn Value'], axis=1) 
y = df['Churn Value'] # The numerical target (0 or 1)

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
# Exclude 'customerID' if it's still in X
categorical_features = X.select_dtypes(include=['object']).columns

# Apply Standard Scaler to numerical data and One-Hot Encoding to categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' # Drops any columns not specified
)




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Use stratify=y to ensure train/test sets have the same Churn proportion.



# Apply transformations only on the features (X)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)

# Get predictions and probabilities for evaluation
y_pred = rf_model.predict(X_test_processed)
y_proba = rf_model.predict_proba(X_test_processed)[:, 1] # Probability of Churn (class 1)



print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Calculate and print AUC-ROC Score
auc_roc = roc_auc_score(y_test, y_proba)
print(f"\nAUC-ROC Score: {auc_roc:.4f}")


fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='No Skill (AUC = 0.5)') # Baseline
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Get feature importance from the trained model
importances = rf_model.feature_importances_

# Create a DataFrame for easy sorting and plotting
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

# Plot the top 10 features (Matplotlib/Seaborn)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Top 10 Feature Importances for Churn Prediction')
plt.show()

print("\nTop 5 Churn Drivers:")
print(feature_importance_df.head(5)['Feature'].tolist())