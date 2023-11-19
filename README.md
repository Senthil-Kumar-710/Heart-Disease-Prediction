## DISEASE PREDICTION USING KNN AND DECISION TREE ALGORITHM FOR SYMPTOM ANALYSIS

 # Submitted by 
# Senthil Kumar S 212221230091 
# Pavan Kishore M 212221230076 

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/pavankishore-AIDS/disease-prediction/blob/main/Mini_Project_Final.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "2R8uTbscMaCQ"
      },
      "outputs": [],
      "source": [
        "# Import libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Read CSV file\n",
        "data = pd.read_csv('heart.csv')\n",
        "\n",
        "# Pre-process the data\n",
        "data.dropna(inplace=True)"
      ],
      "metadata": {
        "id": "eP_EsivMfIs2"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Separate features and target variable\n",
        "X = data.drop('target', axis=1)\n",
        "y = data['target']\n",
        "\n",
        "# Split the data into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
      ],
      "metadata": {
        "id": "rbyelbpBaVNc"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)"
      ],
      "metadata": {
        "id": "zUVEEWPHfhKl"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Train KNN model\n",
        "knn_model = KNeighborsClassifier(n_neighbors=5)\n",
        "knn_model.fit(X_train_scaled, y_train)\n",
        "\n",
        "# Predictions\n",
        "knn_predictions = knn_model.predict(X_test_scaled)\n",
        "\n",
        "# Evaluate the model\n",
        "knn_accuracy = accuracy_score(y_test, knn_predictions)\n",
        "print(f'KNN Accuracy: {knn_accuracy}')\n",
        "print('Classification Report:\\n', classification_report(y_test, knn_predictions))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4eWGrfBB9PSi",
        "outputId": "01647955-70f0-4935-9c84-a25e94ac3d58"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "KNN Accuracy: 0.8341463414634146\n",
            "Classification Report:\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "           0       0.88      0.77      0.82       102\n",
            "           1       0.80      0.89      0.84       103\n",
            "\n",
            "    accuracy                           0.83       205\n",
            "   macro avg       0.84      0.83      0.83       205\n",
            "weighted avg       0.84      0.83      0.83       205\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Train Decision Tree model\n",
        "dt_model = DecisionTreeClassifier(random_state=42)\n",
        "dt_model.fit(X_train, y_train)\n",
        "\n",
        "# Predictions\n",
        "dt_predictions = dt_model.predict(X_test)\n",
        "\n",
        "# Evaluate the model\n",
        "dt_accuracy = accuracy_score(y_test, dt_predictions)\n",
        "print(f'Decision Tree Accuracy: {dt_accuracy}')\n",
        "print('Classification Report:\\n', classification_report(y_test, dt_predictions))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w9sQPz909RJS",
        "outputId": "78b3619f-f24e-4935-e31b-da550a8770e1"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Decision Tree Accuracy: 0.9853658536585366\n",
            "Classification Report:\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "           0       0.97      1.00      0.99       102\n",
            "           1       1.00      0.97      0.99       103\n",
            "\n",
            "    accuracy                           0.99       205\n",
            "   macro avg       0.99      0.99      0.99       205\n",
            "weighted avg       0.99      0.99      0.99       205\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Compare Models\n",
        "print(f'KNN Accuracy: {knn_accuracy}')\n",
        "print(f'Decision Tree Accuracy: {dt_accuracy}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OJ29PaNM9VwB",
        "outputId": "220e3a94-5573-40ce-a5c5-0902779e3d70"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "KNN Accuracy: 0.8341463414634146\n",
            "Decision Tree Accuracy: 0.9853658536585366\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Create a Random Forest using KNN and Decision Tree as base estimators\n",
        "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "rf_model.fit(X_train, y_train)\n",
        "\n",
        "# Predictions\n",
        "rf_predictions = rf_model.predict(X_test)\n",
        "\n",
        "# Evaluate the model\n",
        "rf_accuracy = accuracy_score(y_test, rf_predictions)\n",
        "print(f'Random Forest Accuracy: {rf_accuracy}')\n",
        "print('Classification Report:\\n', classification_report(y_test, rf_predictions))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RRLtgmz99bNV",
        "outputId": "b056c707-2009-437c-d76a-f6abb74ce94f"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Random Forest Accuracy: 0.9853658536585366\n",
            "Classification Report:\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "           0       0.97      1.00      0.99       102\n",
            "           1       1.00      0.97      0.99       103\n",
            "\n",
            "    accuracy                           0.99       205\n",
            "   macro avg       0.99      0.99      0.99       205\n",
            "weighted avg       0.99      0.99      0.99       205\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to take user input for feature values\n",
        "def get_user_input():\n",
        "    age = float(input(\"Enter age: \"))\n",
        "    sex = float(input(\"Enter sex (0 for female, 1 for male): \"))\n",
        "    cp = float(input(\"Enter chest pain type: \"))\n",
        "    trestbps = float(input(\"Enter resting blood pressure: \"))\n",
        "    chol = float(input(\"Enter serum cholesterol: \"))\n",
        "    fbs = float(input(\"Enter fasting blood sugar (0 if < 120 mg/dl, 1 if >= 120 mg/dl): \"))\n",
        "    restecg = float(input(\"Enter resting electrocardiographic results: \"))\n",
        "    thalach = float(input(\"Enter maximum heart rate achieved: \"))\n",
        "    exang = float(input(\"Enter exercise-induced angina (0 for no, 1 for yes): \"))\n",
        "    oldpeak = float(input(\"Enter ST depression induced by exercise relative to rest: \"))\n",
        "    slope = float(input(\"Enter the slope of the peak exercise ST segment: \"))\n",
        "    ca = float(input(\"Enter number of major vessels colored by fluoroscopy: \"))\n",
        "    thal = float(input(\"Enter thalassemia type: \"))\n",
        "\n",
        "    user_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)\n",
        "\n",
        "    return user_data\n",
        "\n",
        "# Get user input\n",
        "user_data = get_user_input()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_SRCs6iO23LM",
        "outputId": "36077632-ce9c-43cc-c26f-0b57797e8061"
      },
      "execution_count": 12,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Enter age: 56\n",
            "Enter sex (0 for female, 1 for male): 0\n",
            "Enter chest pain type: 1\n",
            "Enter resting blood pressure: 140\n",
            "Enter serum cholesterol: 204\n",
            "Enter fasting blood sugar (0 if < 120 mg/dl, 1 if >= 120 mg/dl): 0\n",
            "Enter resting electrocardiographic results: 0\n",
            "Enter maximum heart rate achieved: 109\n",
            "Enter exercise-induced angina (0 for no, 1 for yes): 0\n",
            "Enter ST depression induced by exercise relative to rest: 0\n",
            "Enter the slope of the peak exercise ST segment: 1\n",
            "Enter number of major vessels colored by fluoroscopy: 2\n",
            "Enter thalassemia type: 2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Make knn predictions\n",
        "knn_prediction = knn_model.predict(user_data)\n",
        "print(knn_prediction)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-s2ZA7_KfH2L",
        "outputId": "a204d87e-f60f-41ee-ef91-438850b95294"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Print the knn result\n",
        "if knn_prediction == 1:\n",
        "  print(\"Sorry, You have heart disease\")\n",
        "else:\n",
        "  print(\"Congratulations, You do not have heart disease\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AnkT4MkR8_gN",
        "outputId": "bbcdef78-5ed1-42b3-8155-9db8023cdd5d"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Sorry, You have heart disease\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Make dt predictions\n",
        "dt_prediction = dt_model.predict(user_data)\n",
        "print(dt_prediction)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FXXPCARgUGqJ",
        "outputId": "dedf2a67-404b-425b-8322-9b68de49b1a6"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Print the dt result\n",
        "if dt_prediction == 1:\n",
        "  print(\"Sorry, You have heart disease\")\n",
        "else:\n",
        "  print(\"Congratulations, You do not have heart disease\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1_WjUmYkfQra",
        "outputId": "a2bd1379-1814-4f64-d9f3-c56ea4621f0f"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Congratulations, You do not have heart disease\n"
          ]
        }
      ]
    }
  ]
}```

```
