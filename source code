import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = {
    "age": [29, 45, 31, 50, 62, 35, 42, 55],
    "weight": [65, 85, 70, 90, 72, 67, 89, 74],
    "blood_pressure": [120, 140, 130, 150, 135, 128, 145, 138],
    "glucose": [80, 150, 100, 170, 110, 95, 160, 120],
    "family_history": [0, 1, 0, 1, 1, 0, 1, 0],  # 1 = yes, 0 = no
    "diabetes": [0, 1, 0, 1, 1, 0, 1, 0]  # 1 = diabetic, 0 = no diabetes
}
df = pd.DataFrame(data)

features = ["age", "weight", "blood_pressure", "glucose", "family_history"]
X = df[features]
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
print("Predicted classes for test data:", y_pred)
print("Actual classes for test data:   ", list(y_test))
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

new_user = pd.DataFrame({
    "age": [40],
    "weight": [68],
    "blood_pressure": [135],
    "glucose": [115],
    "family_history": [1]
})
risk = model.predict(new_user)
if risk[0] == 1:
    print("Warning: You might be at high risk for diabetes. Please consult a doctor.")
else:
    print("Your predicted risk for diabetes is low based on these features.")
