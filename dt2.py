import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    try:
        data = pd.read_excel(r"C:\Users\Pratyasha\OneDrive\Documents\ckd_dataset2.xlsx")
        print("Column names in the dataset:", data.columns.tolist())
        return data
    except FileNotFoundError:
        print("The specified file does not exist.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def preprocess_data(data):
    # Check for a possible target column
    possible_targets = ['target', 'class', 'ckd', 'outcome', 'label']
    target_column = None
    for col in data.columns:
        if col.strip().lower() in possible_targets:
            target_column = col
            break
    
    if not target_column:
        print("Error: No valid target column found in dataset.")
        return None, None
    
    print(f"Using '{target_column}' as the target column.")
    
    # Extract features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Handle missing values by filling them with mean for numerical and mode for categorical
    X = X.fillna(X.mean())
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Label Encoding for categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y

def train_decision_tree(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Optional Feature Scaling (only for numerical data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize the Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    
    # Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")
    
    return model, cm, X.columns

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No CKD', 'Predicted CKD'], 
                yticklabels=['Actual No CKD', 'Actual CKD'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=['No CKD', 'CKD'], filled=True, rounded=True)
    plt.title('Decision Tree Visualization')
    plt.show()

# Use hardcoded file path for testing
file_path = r"C:\Users\Pratyasha\OneDrive\Documents\ckd_dataset2.xlsx"
data = load_data(file_path)

if data is not None:
    X, y = preprocess_data(data)
    if X is not None and y is not None:
        print("\nData preprocessing successful!")
        print("Training Decision Tree model...")
        
        # Train and evaluate the model
        model, cm, feature_names = train_decision_tree(X, y)
        
        #Decision Tree
        print("\nVisualizing the Decision Tree...")
        plot_decision_tree(model, feature_names)
        
        # Confusion Matrix
        print("\nVisualizing the Confusion Matrix...")
        plot_confusion_matrix(cm)
    else:
        print("Data preprocessing failed.")
else:
    print("Failed to load dataset.")
