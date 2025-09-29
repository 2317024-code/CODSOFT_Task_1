# Titanic Survival Prediction  

## Overview  
This project predicts **passenger survival on the Titanic** using a **Logistic Regression model**. By analyzing passenger details such as age, sex, class, family size, and title, the model learns patterns that determine survival likelihood. The project includes data preprocessing, feature engineering, model training, evaluation, and prediction.  

## Features  
- Load and clean the Titanic dataset  
- Handle missing values in **Age** and **Embarked**  
- Drop irrelevant columns such as **Cabin, Ticket, PassengerId, Name**  
- Feature engineering:  
  - **FamilySize** → sum of siblings/spouses and parents/children + 1  
  - **IsAlone** → indicator for passengers traveling alone  
  - **Title** → extracted from passenger names  
- Encode categorical variables using **one-hot encoding**  
- Split dataset into training and testing sets  
- Train a **Logistic Regression model**  
- Evaluate model performance using:  
  - Accuracy Score  
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-score)  
- Display sample predictions  

## Dataset  
- Dataset used: `Titanic-Dataset (2).csv`  
- Key Columns:  
  - **PassengerId** → Passenger identifier  
  - **Survived** → Target variable (0 = Did not survive, 1 = Survived)  
  - **Pclass** → Passenger class  
  - **Name** → Passenger name  
  - **Sex** → Gender  
  - **Age** → Age in years  
  - **SibSp** → Number of siblings/spouses aboard  
  - **Parch** → Number of parents/children aboard  
  - **Ticket, Fare, Cabin, Embarked** → Additional passenger details  

- Preprocessing steps:  
  - Fill missing **Age** with median  
  - Fill missing **Embarked** with mode  
  - Drop **Cabin** due to high missing values  
  - Convert categorical variables to numeric via one-hot encoding  

## Technologies Used  
- Python  
- Pandas  
- NumPy  
- Scikit-learn
- VS Code 
