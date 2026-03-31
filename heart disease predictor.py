import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# First, I need to define the column names because the raw file doesn't have them
cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

# Loading the Cleveland dataset we unzipped earlier
# I noticed some '?' in the data, so I'm telling pandas to treat them as NaN
df = pd.read_csv('processed.cleveland.data', names=cols, na_values="?")

# Handling missing values by filling them with the mean of the column
# This is better than just deleting the rows
df = df.fillna(df.mean())

# The target column has values 0-4, but for basic ML we just need 0 or 1
# 0 = healthy, anything else (1,2,3,4) = heart disease
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Checking if the data looks okay now
print("Checking first few rows:")
print(df.head())

# Splitting the data into X (features) and y (what we want to predict)
X = df.drop('target', axis=1)
y = df['target']

# Doing an 80-20 split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the Random Forest model
# I used 100 trees to keep it stable
my_model = RandomForestClassifier(n_estimators=100)
my_model.fit(X_train, y_train)

# Testing it on the 20% we held back
predictions = my_model.predict(X_test)

# Printing the final accuracy
final_acc = accuracy_score(y_test, predictions)
print(f"\nFinal Accuracy is: {final_acc * 100:.2f}%")

# Making a quick chart to see the results
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap='Greens')
plt.title('My Model Results')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()
