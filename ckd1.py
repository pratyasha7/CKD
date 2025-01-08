import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
try:
    data = pd.read_excel(r"C:\Users\Pratyasha\OneDrive\Documents\idp_ckd\ckd_dataset1.xlsx")
except FileNotFoundError:
    print("The specified file was not found.")
    exit()

# Data Exploration
print("First few rows of the dataset:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Check for the 'classification' column
if 'classification' not in data.columns:
    print("Error: 'classification' column not found in the dataset.")
    print("Available columns:", data.columns.tolist())
    exit()

# Identify numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Handling Missing Values for Numeric Columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Convert categorical variables to numeric using Label Encoding
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Convert to string to handle any non-numeric values

# Preprocessing
# Use 'classification' as the target variable
X = data.drop('classification', axis=1)  # Features
y = data['classification']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning using Grid Search
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']
print(f"\nBest k found: {best_k}")

# Initialize the KNN classifier with the best k
knn = KNeighborsClassifier(n_neighbors=best_k)

# Fit the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-Validation
cv_scores = cross_val_score(knn, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Example of predicting a new instance
# Ensure that new_instance has the same features as X
# Replace the values below with actual feature values
new_instance = [[1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]]  # Example feature values
new_instance = pd.DataFrame(new_instance, columns=X.columns)  # Align features
new_instance_scaled = scaler.transform(new_instance)
prediction = knn.predict(new_instance_scaled)
print("\nPrediction for new instance (1 = CKD, 0 = No CKD):", prediction[0])