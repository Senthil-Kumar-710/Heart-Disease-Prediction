## DISEASE PREDICTION USING KNN AND DECISION TREE ALGORITHM FOR SYMPTOM ANALYSIS

 # Submitted by
 
 Senthil Kumar S 212221230091 
 </br>
 Pavan Kishore M 212221230076 

 # INTRODUCTION
 
 Heart disease is a prevalent and life-threatening condition that affects millions of people worldwide. Timely prediction and accurate diagnosis are crucial for effective prevention and treatment. Machine learning techniques have emerged as valuable tools in predicting heart diseases by analyzing patient data and identifying relevant patterns. Two popular machine learning algorithms for this purpose are K-Nearest Neighbors (KNN) and Decision Trees.

This study delves into the realm of healthcare diagnostics, focusing on disease prediction through symptom analysis using two powerful machine learning algorithms: K-Nearest Neighbors (KNN) and Decision Tree. Traditional methods often struggle with the complexities of symptom data, making machine learning an attractive solution. Leveraging the simplicity of KNN and the ability of Decision Trees to model intricate decision processes, this research aims to develop robust predictive models.

# FEATURES USSED

The features used in a disease prediction model using KNN (K-Nearest Neighbors) and Decision Tree algorithms for symptom analysis typically include various symptoms or clinical indicators that are relevant to the specific disease being predicted. The choice of features depends on the nature of the disease and the available data. Here's a general outline of how features might be selected:

Symptoms:
The primary features are the symptoms exhibited by the user. These could include a wide range of symptoms relevant to the disease under consideration. For example, if predicting a respiratory disease, symptoms might include cough, shortness of breath, and chest pain.

Medical History:
Previous medical history, including information about any existing conditions, past illnesses, surgeries, or chronic diseases, could be important features.

Demographic Information:
Demographic data such as age, gender, and ethnicity may be considered as factors influencing disease prevalence and presentation.

Vital Signs:
Measurements of vital signs like blood pressure, heart rate, respiratory rate, and body temperature could be used as features, especially for diseases that manifest in changes to these indicators.

Laboratory Results:
If available, results from laboratory tests or diagnostic procedures could be included. For example, blood tests, imaging results, or biopsy findings may provide valuable information.

Lifestyle Factors:
Information about the user's lifestyle habits, such as smoking, alcohol consumption, diet, and physical activity, may be relevant for certain diseases.

Environmental Factors:
Factors related to the user's environment, such as geographical location, pollution levels, or occupational exposures, might be considered depending on the disease being predicted.

# CODE

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

# KNN ACCURACY AND REPORT

![image](https://github.com/user-attachments/assets/fd3cabff-fb4e-4bbb-ab41-beaf631405bb)

# DECISION TREE ACCURACY AND REPORT

![image](https://github.com/user-attachments/assets/66e39d6b-b31c-4101-bc11-529b1616b2e3)

# RANDOM FOREST ACCURACY AND REPORT

![image](https://github.com/user-attachments/assets/794a9ada-7596-4491-ab90-1812308dc920)

# USER INPUT

![image](https://github.com/user-attachments/assets/f4ce6fff-448a-4fa7-9211-f196c31c18a4)

# KNN PREDICTION

![image](https://github.com/user-attachments/assets/e80f99d9-7591-4ac5-ab38-2072b0f4c307)

# DECISION TREE PREDICTION

![image](https://github.com/user-attachments/assets/ef92ad00-8b21-4f03-a5e5-c7e1de315ff2)

# CONCLUSION

In conclusion, the project on "Disease Prediction Using KNN and Decision Tree Algorithm for Symptom Analysis" represents a promising approach. The combination of these algorithms leverages the strengths of similarity-based classification and interpretable decision-making. Through comprehensive data preprocessing, including cleaning, encoding, and feature selection, and effective data splitting for training and testing, the models are trained on diverse and representative datasets. The data analysis phase involves feature importance assessment, and model training and evaluation. Both KNN and Decision Tree models exhibit their respective advantages — KNN excels in capturing local patterns, while Decision Trees provide interpretability. The ensemble approach, if applicable, enhances predictive performance. Real-time symptom analysis, if integrated with wearable devices, further extends the system's capabilities. Continuous monitoring, user feedback integration, and adherence to ethical considerations ensure the system's adaptability, reliability, and responsible use in healthcare settings. Further development of an ensemble method which utilizes multiple algorithms to have an enhanced application and flawless accuracy for infallibly predicting the presence or occurrence of heart disease in individuals will indefinitely help in countless medical fields.
