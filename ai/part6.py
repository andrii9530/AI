import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error


df = pd.read_csv('internship_candidates_final_numeric.csv')  

features = ['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']
X = df[features]
y = df['Accepted']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] 


plt.figure(figsize=(10, 6))
sc = plt.scatter(X_test['EntryTestScore'], X_test['EnglishLevel'], c=y_proba, cmap='coolwarm', edgecolor='k', s=100)
plt.xlabel("Entry Test Score")
plt.ylabel("English Level")
plt.title("Ймовірність прийняття на стажування (логістична регресія)")
plt.colorbar(sc, label='Ймовірність прийняття')
plt.grid(True)
plt.show()
