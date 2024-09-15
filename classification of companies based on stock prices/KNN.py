#updated KNN code 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
data = pd.read_csv(r"C:\Users\Deeksha\Desktop\DM\financialdata_aka.csv", encoding='latin1')


label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

# Extract features and target variable
X = data.drop('class', axis=1)  # Features
y = data['class']  # Target variable
onehot_encoder = OneHotEncoder()
company_encoded = onehot_encoder.fit_transform(X[['Companies']])

# Create a DataFrame with the encoded "Companies" column
company_encoded_df = pd.DataFrame(company_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(['Companies']))

X_encoded = pd.concat([X.drop(columns=['Companies']), company_encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.1, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn = KNeighborsClassifier()


y_pred = cross_val_predict(knn, X_train_scaled, y_train, cv=5)
conf_matrix = confusion_matrix(y_train, y_pred)


accuracy = accuracy_score(y_train, y_pred) * 100
print("Accuracy:", accuracy)

# Print confusion matrix
print("Confusion Matrix:\n", conf_matrix)


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


cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=15)

# Print average accuracy across 10 folds
avg_cv_accuracy = np.mean(cv_scores) * 100
print("Average Cross-Validation Accuracy:", avg_cv_accuracy)
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])  # TP / (TP + FP)
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])     # TP / (TP + FN)

print("Precision:", precision)
print("Recall:", recall)

error_rate = 100 - accuracy
plt.figure(figsize=(6, 5))
plt.bar('Accuracy', accuracy, color='blue', alpha=0.7, label='Accuracy') # Plot accuracy
plt.bar('Error Rate', error_rate, color='red', alpha=0.7, label='Error Rate') # Plot error rate
plt.title('Accuracy and Error Rate')
plt.ylabel('Percentage')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.show()