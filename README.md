# CancerCellDetection 
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('path_to_dataset.csv')  # Replace 'path_to_dataset.csv' with the actual path

# Preprocessing
X = np.array(data.drop('label', axis=1))
y = np.array(data['label'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVC()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)
