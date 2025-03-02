import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
try:
    data = pd.read_excel(r"C:\Users\Pratyasha\OneDrive\Documents\idp_ckd\ckd_dataset2.xlsx")
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

# Check for the 'Class' column
if 'Class' not in data.columns:
    print("Error: 'Class' column not found in the dataset.")
    print("Available columns:", data.columns.tolist())
    exit()

# Identify numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Handling Missing Values for Numeric Columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Convert categorical variables to numeric using Label Encoding
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}  # Dictionary to store label encoders for each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Convert to string to handle any non-numeric values
    label_encoders[col] = le  # Save the encoder for future use

# Preprocessing
# Assume 'Class' is the target variable where 1 indicates CKD and 0 indicates no CKD
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes of the datasets
print(f"Shape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()

# Fit the scaler only on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Print shapes after scaling
print(f"Shape of scaled X_train: {X_train_scaled.shape}, Shape of scaled X_test: {X_test_scaled.shape}")

# Hyperparameter Tuning using Grid Search
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_k = grid_search.best_params_['n_neighbors']
print(f"\nBest k found: {best_k}")

# Initialize the KNN classifier with the best k
knn = KNeighborsClassifier(n_neighbors=best_k)

# Fit the model
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the KNN model: {accuracy * 100:.2f}%")

# Cross-Validation
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")

# Visualize the Confusion Matrix and save it as an image
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Adjusting the new instance to match the feature count of the training data
def adjust_new_instance(features, template):
    """Adjusts a new instance to match the number of features in the training set."""
    if len(features[0]) != len(template.columns):
        # If length doesn't match, pad with zeros or trim
        adjusted_instance = np.zeros((1, len(template.columns)))
        adjusted_instance[0, :len(features[0])] = features[0]
        return adjusted_instance
    return features

# Example of predicting a new instance
# Replace the values below with actual feature values
new_instance = [[1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]]  # Example feature values

# Adjust new instance to match the training feature count
new_instance_adjusted = adjust_new_instance(new_instance, X)
new_instance_scaled = scaler.transform(new_instance_adjusted)
prediction = knn.predict(new_instance_scaled)
print("\nPrediction for new instance (1 = CKD, 0 = No CKD):", prediction[0])
