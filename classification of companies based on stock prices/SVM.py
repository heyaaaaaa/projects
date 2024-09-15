
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv(r"D:/coding/python/Data_Mining/project/updated_data.csv")

# Separate features (X) and target variable (y)
X = data.drop(columns=['class', 'Companies'])
y = data['class']

# Apply SMOTE oversampling to the data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Define the parameter grid for grid search with additional kernel options
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'poly', 'rbf']}

# Initialize the SVM classifier
svm_classifier = SVC(random_state=42)

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, y_resampled)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the SVM classifier with the best parameters
best_svm_classifier = SVC(**best_params, random_state=42)

# Initialize KFold with 3 folds
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Lists to store accuracy and error rate 
accuracies = []
error_rates = []

# Initialize the confusion matrix
conf_matrix = np.zeros((3, 3))

for train_index, test_index in kf.split(X_scaled):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y_resampled[train_index], y_resampled[test_index]

    best_svm_classifier.fit(X_train_fold, y_train_fold)
    y_pred_fold = best_svm_classifier.predict(X_test_fold)

    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold) * 100
    error_fold = 100 - accuracy_fold

    # Update the confusion matrix
    conf_matrix += confusion_matrix(y_test_fold, y_pred_fold)

    accuracies.append(accuracy_fold)
    error_rates.append(error_fold)

# Calculate average accuracy and error rate
avg_accuracy = np.mean(accuracies)
avg_error_rate = np.mean(error_rates)

print("Average Accuracy:", round(avg_accuracy))
print("Average Error Rate:", round(avg_error_rate))

# Print classification report
print(classification_report(y_test_fold, y_pred_fold))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Print cross-validation scores
print("Cross-Validation Scores:", accuracies)
print("Mean Accuracy after validation:", round(avg_accuracy))

# Plot accuracy and error rate
plt.figure(figsize=(4, 4))
plt.bar(['Accuracy', 'Error Rate'], [avg_accuracy, avg_error_rate], color=['green', 'red'])
plt.ylim(0, 100)
plt.ylabel('Percentage')
plt.title('Accuracy and Error Rate')
plt.show()