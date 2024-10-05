# Breast Cancer Prediction Model

This project involves building a predictive model to classify whether a patient has breast cancer based on various clinical features. The dataset used for training and evaluation includes several medical attributes related to tumor characteristics, and the goal is to predict if the tumor is malignant or benign.

## Project Overview

In this project, we explore multiple machine learning algorithms to identify which one performs best for predicting breast cancer outcomes. The following algorithms were implemented and evaluated:

- **Logistic Regression**
- **Support Vector Classifire (SVC)**
- **Random Forest Classifire**
- **Gradient Boosting**

After thorough evaluation, **Logistic Regression** achieved the highest accuracy, demonstrating strong performance in predicting the correct diagnosis based on the input features.

## Key Features of the Project:

- **Data Preprocessing**: Handling missing values, scaling the data, and splitting it into training and testing sets.
- **Model Training and Evaluation**: Multiple algorithms are trained, and their performance is compared using metrics such as accuracy, precision, recall, and F1-score.
- **Highest Accuracy with Logistic Regression**: Among the different models, Logistic Regression provided the best accuracy, making it the final model of choice for this project.

## Objective

The primary objective of this project is to develop a robust model that can assist medical professionals in diagnosing breast cancer with high accuracy, potentially leading to faster and more reliable treatment decisions.



# Dataset
```
from sklearn.datasets import load_breast_cancer
```

```
data = load_breast_cancer()
```

## About dataset
```
print(data.DESCR)
```

💿 Installing
1. Environment setup.
```
conda create --prefix venv python==3.8 -y
```
```
conda activate /.venv
````
2. Install Requirements and setup
```
pip install -r requirements.txt
```
5. Run Application
```
python app.py
```
## for prediction go to 
http://127.0.0.1:5000/predict

## My learnings 

  - Machine learning model to predict breast cancer malignancy with 95% accuracy using Logistic Regression. 
  - I gained hands-on experience in file structures, pipelines, and classification problem-solving.
  - Utilized Scikit-learn for modeling, Flask for UI, and MongoDB for data management.
