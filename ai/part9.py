import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_passenger_ids = test_df['PassengerId']


combined = pd.concat([train_df, test_df], sort=False)



combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
combined['Title'] = combined['Title'].replace(['Mlle', 'Ms'], 'Miss')
combined['Title'] = combined['Title'].replace(['Mme'], 'Mrs')
combined['Title'] = combined['Title'].replace(['Dr', 'Major', 'Col', 'Rev', 'Sir', 'Lady', 'Don', 'Capt', 'Countess', 'Jonkheer'], 'Rare')


combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)


combined['Age'] = combined['Age'].fillna(combined['Age'].median())
combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])


label_encoders = {}
for col in ['Sex', 'Embarked', 'Title']:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le


combined.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


train_size = train_df.shape[0]
train_data = combined[:train_size]
test_data = combined[train_size:]


X = train_data.drop(['Survived', 'PassengerId'], axis=1)
y = train_df['Survived']
X_test = test_data.drop(['Survived', 'PassengerId'], axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print(" Точність на валідаційному наборі:", accuracy_score(y_val, y_pred))
print(" Звіт класифікації:\n", classification_report(y_val, y_pred))


test_predictions = model.predict(X_test_scaled)
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': test_predictions
})
submission.to_csv('titanic_submission.csv', index=False)
print(" Файл 'titanic_submission.csv' успішно збережено.")
