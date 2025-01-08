# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_excel(r"C:\Users\Pratyasha\OneDrive\Documents\ckd_dataset2.xlsx")
# Display dataset overview
print("Dataset Overview:")
print(data.head())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Handle missing values (fill with median for numerical columns)
data.fillna(data.median(numeric_only=True), inplace=True)

# Encode the target variable if it's categorical
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

# Separate features (X) and target (y)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply PCA to reduce the data to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split PCA-transformed data into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train the SVM model on the PCA-transformed data
model = SVC(kernel='linear', random_state=42)
model.fit(X_train_pca, y_train_pca)

# Evaluate the model
y_pred_pca = model.predict(X_test_pca)
accuracy = accuracy_score(y_test_pca, y_pred_pca)
print(f"Accuracy on PCA-transformed data: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test_pca, y_pred_pca)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test_pca, y_pred_pca))

# Visualize the decision boundary in 2D
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the classes for each point in the mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and data points
plt.figure(figsize=(12, 6))

# Subplot 1: Decision Boundary
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("SVM Decision Boundary (2D PCA-transformed data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# Subplot 2: Confusion Matrix
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Show plots
plt.tight_layout()
plt.show()