
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

class Node:
    def __init__(self, feature=None, value=None, results=None, left=None, right=None):
        self.feature = feature  
        self.value = value  
        self.results = results  
        self.left = left  
        self.right = right  

def entropy(data):
    label_counts = Counter(data)
    entropy = 0.0
    total_instances = len(data)
    for label in label_counts:
        prob = label_counts[label] / total_instances
        entropy -= prob * np.log2(prob)
    return entropy

def information_gain(left, right):
    p = float(len(left)) / (len(left) + len(right))
    return entropy(left + right) - p * entropy(left) - (1 - p) * entropy(right)

def split_data(data, feature_index, value):
    left, right = [], []
    for row in data:
        if row[feature_index] <= value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def find_best_split(data):
    best_gain = 0
    best_feature = None
    best_value = None
    n_features = len(data[0]) - 1  # Exclude the class label column
    for feature_index in range(n_features):
        feature_values = set(row[feature_index] for row in data)
        for value in feature_values:
            left, right = split_data(data, feature_index, value)
            gain = information_gain([row[-1] for row in left], [row[-1] for row in right])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_value = value
    return best_feature, best_value

def build_tree(data):
    if len(set(row[-1] for row in data)) == 1:  # If all instances have the same class label
        return Node(results=Counter(row[-1] for row in data))
    
    best_feature, best_value = find_best_split(data)
    if best_feature is None:
        return Node(results=Counter(row[-1] for row in data))
    
    left_data, right_data = split_data(data, best_feature, best_value)
    
    left_subtree = build_tree(left_data)
    right_subtree = build_tree(right_data)
    
    return Node(feature=best_feature, value=best_value, left=left_subtree, right=right_subtree)

def predict(node, instance):
    if node.results is not None:
        return node.results.most_common(1)[0][0]
    if instance[node.feature] <= node.value:
        return predict(node.left, instance)
    else:
        return predict(node.right, instance)

def decision_tree_classification(train_data, test_data):
    tree = build_tree(train_data)
    predictions = [predict(tree, instance) for instance in test_data]
    return predictions

# Load the original dataset
original_dataset = pd.read_csv(r"D:/coding/python/Data_Mining/project/finance_data.csv", encoding='latin1')

class_mapping = {'high': 1, 'mod': 2, 'low': 3}

original_dataset['class'] = original_dataset['class'].map(class_mapping)

X = original_dataset.drop(columns=['class'])
y = original_dataset['class']

# Initialize KFold with 3 folds
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Lists to store accuracy and error rate 
accuracies = []
error_rates = []

# Initialize the confusion matrix
conf_matrix = np.zeros((3, 3))

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    train_data = np.column_stack((X_train, y_train))
    predictions = decision_tree_classification(train_data, np.column_stack((X_test, y_test)))

    accuracy = np.mean(predictions == y_test) * 100
    error_rate = 100 - accuracy

    # Update the confusion matrix
    conf_matrix += confusion_matrix(y_test, predictions)

    accuracies.append(accuracy)
    error_rates.append(error_rate)

# Calculate average accuracy and error rate
avg_accuracy = np.mean(accuracies)
avg_error_rate = np.mean(error_rates)

print("Average Accuracy:", round(avg_accuracy))
print("Average Error Rate:", round(avg_error_rate))

# Print classification report
print("Classification Report:\n", classification_report(y_test, predictions))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['High', 'Moderate', 'Low']
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

plt.figure(figsize=(6, 5))
plt.bar('Accuracy', avg_accuracy, color='blue', alpha=0.7, label='Average Accuracy') # Plot average accuracy
plt.bar('Error Rate', avg_error_rate, color='red', alpha=0.7, label='Average Error Rate') # Plot average error rate
plt.title('Average Accuracy and Error Rate over 15 Folds')
plt.ylabel('Percentage')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.show()