# Hello, welcome to my repository where you learn about the Heart-Disease-Prediction-Machine!!
SO LET'S GOOO!!!

Project Overview
This project is a healthcare-focused AI application developed as a capstone for the Fundamentals of AI and ML course. It uses a clinical dataset to predict whether a patient has a heart condition based on several medical features such as age, cholesterol, and chest pain type.

The goal is to demonstrate the full Machine Learning lifecycle: from raw data cleaning to model evaluation.

Features
Data Cleaning: Automatically handles missing values (?) using mean imputation.

Binary Classification: Simplifies complex medical stages into a clear "Healthy vs. Sick" prediction.

High Accuracy: Uses the Random Forest ensemble algorithm for stable and reliable results.

Visualization: Includes a Confusion Matrix heatmap to visualize model performance.

Tech Stack
Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

Platform: Google Colab

Dataset Information
The model is trained on the UCI Cleveland Heart Disease Dataset.

Total Samples: 303 patients

Attributes: 14 (including age, sex, cp, chol, thalach, etc.)

Target: Binary (0 = No Disease, 1 = Disease Present)

How to Run
Clone this repository or download the files.

Ensure you have the processed.cleveland.data file in the same folder as the script.

Install the required libraries:

Bash
pip install pandas scikit-learn matplotlib seaborn
Run the Python script or open the .ipynb file in Google Colab.

Results
Model Accuracy: 90.16%

Algorithm: Random Forest Classifier (n_estimators=100)

made by:
Vanshika Shankar
