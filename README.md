## DISEASE PREDICTION USING KNN AND DECISION TREE ALGORITHM FOR SYMPTOM ANALYSIS

 # Submitted by 
# Senthil Kumar S 212221230091 
# Pavan Kishore M 212221230076 
```
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Read CSV file
data = pd.read_csv('heart.csv')

# Pre-process the data
data.dropna(inplace=True)
# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predictions
knn_predictions = knn_model.predict(X_test_scaled)

# Evaluate the model
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f'KNN Accuracy: {knn_accuracy}')
print('Classification Report:\n', classification_report(y_test, knn_predictions))
# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
dt_predictions = dt_model.predict(X_test)

# Evaluate the model
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f'Decision Tree Accuracy: {dt_accuracy}')
print('Classification Report:\n', classification_report(y_test, dt_predictions))
# Compare Models
print(f'KNN Accuracy: {knn_accuracy}')
print(f'Decision Tree Accuracy: {dt_accuracy}')
# Create a Random Forest using KNN and Decision Tree as base estimators
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy}')
print('Classification Report:\n', classification_report(y_test, rf_predictions))
# Function to take user input for feature values
def get_user_input():
    age = float(input("Enter age: "))
    sex = float(input("Enter sex (0 for female, 1 for male): "))
    cp = float(input("Enter chest pain type: "))
    trestbps = float(input("Enter resting blood pressure: "))
    chol = float(input("Enter serum cholesterol: "))
    fbs = float(input("Enter fasting blood sugar (0 if < 120 mg/dl, 1 if >= 120 mg/dl): "))
    restecg = float(input("Enter resting electrocardiographic results: "))
    thalach = float(input("Enter maximum heart rate achieved: "))
    exang = float(input("Enter exercise-induced angina (0 for no, 1 for yes): "))
    oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
    slope = float(input("Enter the slope of the peak exercise ST segment: "))
    ca = float(input("Enter number of major vessels colored by fluoroscopy: "))
    thal = float(input("Enter thalassemia type: "))

    user_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

    return user_data

# Get user input
user_data = get_user_input()
# Make knn predictions
knn_prediction = knn_model.predict(user_data)
print(knn_prediction)
# Print the knn result
if knn_prediction == 1:
  print("Sorry, You have heart disease")
else:
  print("Congratulations, You do not have heart disease")
# Make dt predictions
dt_prediction = dt_model.predict(user_data)
print(dt_prediction)
# Print the dt result
if dt_prediction == 1:
  print("Sorry, You have heart disease")
else:
  print("Congratulations, You do not have heart disease")
```
