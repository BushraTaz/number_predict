# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !pip install --upgrade certifi
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import urllib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']


print(mnist.DESCR)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X = X / 255.0

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.shape)
print(y_train.shape)

# +
# Hyperparameter Tuning for Random Forest
param_grid_rf = {'n_estimators': [50, 100, 200],
                 'max_depth': [None, 10, 20],
                 'min_samples_split': [2, 5, 10]}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train_scaled, y_train)

print("Best parameters for Random Forest:", grid_search_rf.best_params_)
best_rf_model = grid_search_rf.best_estimator_
# -

# Evaluate model
print("Random Forest Accuracy:", accuracy_score(y_test, best_rf_model.predict(X_test_scaled)))

from sklearn.metrics import classification_report
# Evaluate models
y_pred_rf = best_rf_model.predict(X_test_scaled)
# Print classification reports
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))


# +
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Plot confusion matrix heatmap using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_rf, annot=True, cmap='Blues', fmt='d', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Random Forest')
plt.show()


# +
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the hyperparameters and their possible values
param_grid_knn = {
    'n_neighbors': [3, 5, 10],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
}

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier()

# Perform grid search with cross-validation
grid_search_knn = GridSearchCV(knn_classifier, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(X_train_scaled, y_train)

# Print the best parameters found by grid search
print("Best parameters for KNN:", grid_search_knn.best_params_)

# Get the best KNN model
best_knn_model = grid_search_knn.best_estimator_

# Evaluate the best KNN model
knn_accuracy = accuracy_score(y_test, best_knn_model.predict(X_test_scaled))
print("KNN Accuracy:", knn_accuracy)


# +
from sklearn.metrics import classification_report

# Evaluate the best KNN model
y_pred_knn = best_knn_model.predict(X_test_scaled)

# Generate classification report
print("Classification Report for KNN:")
print(classification_report(y_test, y_pred_knn))


# +
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate confusion matrix for KNN
knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)


# Plot confusion matrix heatmap using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(knn_confusion_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Knn')
plt.show()

# +
from sklearn.ensemble import VotingClassifier

# Define the tuned models
best_rf_model = grid_search_rf.best_estimator_
best_knn_model = grid_search_knn.best_estimator_

# Create a Voting Classifier with the tuned models
voting_classifier = VotingClassifier(
    estimators=[
        ('random_forest', best_rf_model),
        ('knn', best_knn_model),
    ],
    voting='hard'  # 'hard' for majority voting, 'soft' for weighted voting based on probabilities
)

# Fit the Voting Classifier to the training data
voting_classifier.fit(X_train_scaled, y_train)

# -

# Evaluate the Voting Classifier
voting_accuracy = accuracy_score(y_test, voting_classifier.predict(X_test_scaled))
print("Voting Classifier Accuracy:", voting_accuracy)

# Generate classification report for Voting Classifier
y_pred_voting = voting_classifier.predict(X_test_scaled)
print("Classification Report for Voting Classifier:")
print(classification_report(y_test, y_pred_voting))

# +
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate confusion matrix for Voting Classifier
voting_confusion_matrix = confusion_matrix(y_test, y_pred_voting)

# Plot confusion matrix heatmap using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_rf, annot=True, cmap='Blues', fmt='d', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Voting classifier')
plt.show()
# -

# Evaluate on the test set
test_accuracy = accuracy_score(y_test, voting_classifier.predict(X_test_scaled))
print("Test Accuracy:", test_accuracy)

# +
import joblib
import cv2
import numpy as np
import streamlit as st

# Save the best models
joblib.dump(best_rf_model, 'best_rf_model.pkl')
# Load the RF model
best_rf_model = joblib.load('best_rf_model.pkl')

# Function to preprocess the image
def preprocess_image(image):
    # Resize image to 28x28 pixels
    resized_image = cv2.resize(image, (28, 28))
    # Convert image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Invert the colors (MNIST digits are white on black background)
    inverted_image = cv2.bitwise_not(gray_image)
    # Reshape the image to match MNIST data format (784 features)
    flattened_image = inverted_image.reshape(1, -1)
    # Scale pixel values to be between 0 and 1
    scaled_image = flattened_image / 255.0
    return scaled_image

# Streamlit UI
st.title('Handwritten Digit Recognition')

# Allow user to upload image or capture image from camera
image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if image_file is not None:
    # Read the image
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make prediction
    prediction_rf = best_rf_model.predict(preprocessed_image)
 
    # Display the predictions
    st.write('Random Forest Prediction:', prediction_rf[0])


# Allow user to capture image from camera
if st.button('Capture Image'):
    # Access camera
    video_capture = cv2.VideoCapture(0)
    
    # Capture image
    ret, frame = video_capture.read()
    if ret:
        # Display captured image
        st.image(frame, caption='Captured Image', use_column_width=True)
        
        # Preprocess the captured image
        preprocessed_image = preprocess_image(frame)
        
        # Make prediction
        prediction_rf = best_rf_model.predict(preprocessed_image)
       
        
        # Display the predictions
        st.write('Random Forest Prediction:', prediction_rf[0])
      
    # Release the camera
    video_capture.release()

# -


