# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (assuming it's in a CSV file)
data = pd.read_csv('marketing_data.csv')  # Replace 'marketing_data.csv' with your dataset file path

# Perform Exploratory Data Analysis (EDA)
# EDA involves data cleaning, feature selection, and data exploration

# Data Cleaning: Handle missing values if any
data.dropna(inplace=True)

# Feature Selection: Choose relevant features for the prediction model
selected_features = ['feature1', 'feature2', 'feature3']  # Replace with actual feature names

X = data[selected_features]  # Features
y = data['purchase']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing: Standardize the features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Now, you can use this trained model for predictions on new marketing leads.

# Export the model to be used in IBM Cognos Analytics or another platform
import joblib
joblib.dump(rf_classifier, 'marketing_model.pkl')  # Save the model to a file
