
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
dataset = pd.read_csv(r"C:\Users\abhis\OneDrive\Desktop\finance_data_final.csv", encoding='latin1')

# Drop the first row
dataset = dataset.iloc[1:]

# Convert class labels to numeric values
class_mapping = {'high': 0, 'low': 1, 'mod': 2}
dataset['class'] = dataset['class'].map(class_mapping)

# Label encode categorical features
label_encoder = LabelEncoder()
dataset['Companies'] = label_encoder.fit_transform(dataset['Companies'])

# Separate features and target variable
X = dataset.drop(columns=['class'])
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the class distribution on the training set only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()

# Apply SMOTE to balance the class distribution
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Initialize KFold with 3 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store accuracy and error rate 
accuracies = []
error_rates = []

# Initialize the confusion matrix
conf_matrix = np.zeros((3, 3))

for train_index, test_index in kf.split(X_resampled):
    X_train_fold, X_test_fold = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train_fold, y_test_fold = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    # Train the Naive Bayes classifier
    nb_classifier.fit(X_train_fold, y_train_fold)
    
    # Predict on the test fold
    y_pred_fold = nb_classifier.predict(X_test_fold)

    # Calculate accuracy and error rate for the fold
    accuracy_fold = accuracy_score(y_test_fold, y_pred_fold) * 100
    error_fold = 100 - accuracy_fold

    # Update the confusion matrix
    conf_matrix += confusion_matrix(y_test_fold, y_pred_fold)

    # Append accuracy and error rate to lists
    accuracies.append(accuracy_fold)
    error_rates.append(error_fold)

# Calculate average accuracy and error rate
avg_accuracy = np.mean(accuracies)
avg_error_rate = np.mean(error_rates)

print("Average Accuracy:", round(avg_accuracy, 2))
print("Average Error Rate:", round(avg_error_rate, 2))

# Print confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['High', 'Low', 'Moderate']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], '.0f'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


print("Accuracy after validation:", (avg_accuracy))


# Calculate precision and recall from confusion matrix
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])  # TP / (TP + FP)
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])     # TP / (TP + FN)

print("Precision:", precision)
print("Recall:", recall)

# Plot accuracy and error rate
error_rate = 100 - avg_accuracy
plt.figure(figsize=(6, 5))
plt.bar('Accuracy', avg_accuracy, color='blue', alpha=0.7, label='Accuracy') # Plot accuracy
plt.bar('Error Rate', avg_error_rate, color='red', alpha=0.7, label='Error Rate') # Plot error rate
plt.title('Average Accuracy and Error Rate')
plt.ylabel('Percentage')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.show()