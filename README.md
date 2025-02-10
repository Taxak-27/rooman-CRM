# Improving-data-Accuracy-in-CRM
**Project Description**

This project analyzes customer spending patterns in an online retail dataset to classify customers as High Spenders. Using anomaly detection and a machine learning pipeline, we identify the top 25% of customers by spend and ensure data integrity. This repository includes:

*Data cleaning and preprocessing

*Anomaly detection using Isolation Forest

*Class balancing with SMOTE

*Classification with Random Forest

*Model evaluation and feature importance analysis

**Key Features**

Data Cleaning: Handled missing values and outliers in the dataset.

Feature Engineering: Created new features like Total Spend, Mean Spend Per Transaction, and Transaction Frequency.

Anomaly Detection: Used Isolation Forest to filter out noisy records.

SMOTE Oversampling: Addressed class imbalance for accurate high-spender prediction.

Model Training and Evaluation: Built a Random Forest model with a comprehensive evaluation (classification report, confusion matrix, and feature importance).

Visualization: Heatmaps and bar plots for clear insights.

**Results**

Classification Report: Precision, Recall, F1-score, and Accuracy metrics for high spender prediction.

Confusion Matrix:Visualization of true positives, false positives, true negatives, and false negatives.

Feature Importance:Insights into key predictors for high spenders.

**Future Work**
Add advanced techniques for feature selection and model optimization.

Extend the pipeline to support real-time predictions using live data.

Enhance visualization with interactive dashboards.

Feel free to contribute to this project! Fork the repository and create a pull request for any enhancements or fixes.

Acknowledgments
Dataset source: UCI Machine Learning Repository
Python libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, imblearn
