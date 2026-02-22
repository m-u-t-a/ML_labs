import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/glass.csv")

data = data.drop(columns=["Id"])

X = data.drop(columns=["Type"])
y = data["Type"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=666, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_values = range(1, 31)
errors = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)

    errors.append(error)

plt.figure(figsize=(8, 6))
plt.plot(k_values, errors, marker='o', color = 'green')
plt.xlabel("Number of neighbors (k)")
plt.ylabel("Classif. error")
plt.title("Error(k) dependency")
plt.grid(True)
plt.show()

best_k = k_values[np.argmin(errors)]
print(f"Best k: {best_k}")
print(f"Min mistake: {min(errors):.4f}")