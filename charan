# detecting-fraud-in-financial-transaction
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Load the dataset
data = pd.read_csv("transaction_data.csv")


# Split the dataset into features (X) and labels (y)
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a Random Forest classifier
model = RandomForestClassifier()


# Train the model
model.fit(X_train, y_train)


# Make predictions on the test set
predictions = model.predict(X_test)


# Evaluate the model
print(classification_report(y_test, predictions))
