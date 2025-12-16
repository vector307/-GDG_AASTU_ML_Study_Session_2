# LOGISTIC REGRESSION

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load data
data = pd.read_csv("data.csv")

data.info()
print(data.describe())

# Step 2: Data cleaning
sns.heatmap(data.isnull())
plt.show()

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

# step 3: Encode target variable
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

data["diagnosis"].value_counts().plot(kind="bar")
plt.show()

# Step 4: Split features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 6: Train Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# Step 7: Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
