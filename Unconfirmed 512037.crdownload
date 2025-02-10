import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"D:\Downloads\Project\online_retail_II.xlsx"
data = pd.read_excel(file_path, sheet_name="Year 2009-2010")

print("Initial Dataset Snapshot:")
print(data.head())

print("\nMissing Values Before Cleaning:\n", data.isnull().sum())

data['Customer ID'] = data['Customer ID'].fillna('Unknown')

numeric_cols = ['Quantity', 'Price']
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

categorical_cols = ['Country', 'Description']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing Values After Cleaning:\n", data.isnull().sum())

data['Total_Spend'] = data['Quantity'] * data['Price']
data['Transaction_Frequency'] = data.groupby('Customer ID')['Invoice'].transform('count')
data['Mean_Spend_Per_Transaction'] = data['Total_Spend'] / data['Transaction_Frequency']

data = data[data['Total_Spend'] > 0]

spend_threshold = data['Total_Spend'].quantile(0.75)  # Top 25% as high spenders
data['High_Spender'] = (data['Total_Spend'] >= spend_threshold).astype(int)

print("\nClass Distribution for High Spenders:\n", data['High_Spender'].value_counts())

iso = IsolationForest(contamination=0.01, random_state=42)
data['Anomaly'] = iso.fit_predict(data[['Quantity', 'Price', 'Total_Spend', 'Mean_Spend_Per_Transaction']])

data = data[data['Anomaly'] == 1].drop(columns=['Anomaly'])

X = data[['Quantity', 'Price', 'Total_Spend', 'Mean_Spend_Per_Transaction', 'Transaction_Frequency']]
y = data['High_Spender']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Spender', 'High Spender'],
            yticklabels=['Low Spender', 'High Spender'])
plt.title("Confusion Matrix")
plt.show()

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_}).sort_values(
    by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()

print("Key Observations:")
print("1. Model is trained to predict 'High Spenders' (Top 25% customers by total spend).")
print("2. Features like 'Total Spend' and 'Mean Spend Per Transaction' are critical predictors.")
print("3. Anomaly detection ensures data integrity, removing potentially noisy records.")
