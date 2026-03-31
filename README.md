# Hello, welcome to my repository where you learn about the Heart-Disease-Prediction-Machine!!
SO LET'S GOOO!!!

Heart Disease Diagnostic System:
Project Description
This repository contains a Supervised Machine Learning solution for predicting the presence of heart disease. Using the UCI Cleveland dataset, this project implements a Random Forest Classifier to analyze 14 clinical features and provide a binary classification (Healthy vs. At-Risk).

This project was developed as a capstone for the Fundamentals of AI and ML course, emphasizing the importance of data preprocessing and model evaluation in medical informatics.


The Machine Learning Pipeline
I implemented a robust 5-stage pipeline to ensure the model's reliability:
Data Acquisition: Importing the raw UCI Cleveland .data file.
Preprocessing: Handling missing values (?) via Mean Imputation.
Column mapping and normalization.
Label transformation (Converting multi-class stages 0-4 into binary 0/1).
Exploratory Data Analysis (EDA): Identifying correlations between features like maximum heart rate (thalach) and the target variable.
Model Training: Training a Random Forest Ensemble with 100 individual decision trees.
Evaluation: Using a Confusion Matrix and Accuracy Score to validate results on unseen data.


Feature       Description
Age           Patient age in years
CP            Chest pain type (Value 1-4)
Trestbps      Resting blood pressure
Chol          Serum cholesterol in mg/dl
Thalach       Maximum heart rate achieved
Oldpeak       ST depression induced by exercise


Model Performance
Algorithm: Random Forest Classifier
Test Accuracy: 90.16%
Evaluation Metric: Confusion Matrix (Green Heatmap)
Note: I chose the Random Forest tree over a simple Decision Tree because it reduces Variance and prevents Overfitting, making it safer for medical predictions.


How to Run the Project
1. Clone the Repository: git clone https://github.com/heart-disease-prediction-machine/Health-AI-Capstone.git
2. Upload Data: Ensure processed.cleveland.data is in the root directory.
3. Install Dependencies:pip install pandas numpy scikit-learn seaborn matplotlib
4. Execute: Run the .ipynb file in Google Colab or Jupyter Notebook.


File Structure
1. Heart_Project.ipynb: The main Python notebook containing the cleaned code.
2. processed.cleveland.data: The raw dataset used for training.
3. README.md: Project documentation (this file).


Learning Outcomes
1.I understood the impact of Imputation on model performance.
2.Mastered the Train-Test Split methodology to prevent data leakage.
3.Gained experience in Data Visualization for communicating AI results to non-technical stakeholders.

