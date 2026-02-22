import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/glass.csv")

data = data.drop(columns=["Id"])

X = data.drop(columns=["Type"])
y = data["Type"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=666,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

metrics = {
    "euclidean": {"metric": "euclidean"},
    "manhattan": {"metric": "manhattan"},
    "chebyshev": {"metric": "chebyshev"}
}

k_values = range(1, 31)

plt.figure(figsize=(10, 7))

# for metric_name, params in metrics.items():
#
#     errors = []
#
#     for k in k_values:
#         model = KNeighborsClassifier(n_neighbors=k, **params)
#         model.fit(X_train, y_train)
#
#         y_pred = model.predict(X_test)
#         error = 1 - accuracy_score(y_test, y_pred)
#
#         errors.append(error)
#
#     best_k = k_values[np.argmin(errors)]
#     best_error = min(errors)
#     best_accuracy = 1 - best_error
#
#     print(f"\nMetric: {metric_name}")
#     print(f"Best k: {best_k}")
#     print(f"Best accuracy: {best_accuracy:.4f}")
#
#     plt.plot(k_values, errors, marker='o', label=metric_name)
#
# plt.xlabel("Number of neighbors (k)")
# plt.ylabel("Classif. Error")
# plt.title("Error(k) dependency")
# plt.legend()
# plt.grid(True)
# plt.show()

new_sample = np.array([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])

new_sample_scaled = scaler.transform(new_sample)

model = KNeighborsClassifier(n_neighbors=8, metric="manhattan")
model.fit(X_train, y_train)

predicted_type = model.predict(new_sample_scaled)
print(f"Predicted type: {predicted_type[0]}")
