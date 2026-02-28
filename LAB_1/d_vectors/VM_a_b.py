import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

train_data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/svmdata_a.txt", sep="\t")
test_data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/svmdata_a_test.txt", sep="\t")

label_mapping = {label: idx for idx, label in enumerate(train_data["Color"].unique())}
train_data["Label"] = train_data["Color"].map(label_mapping)
test_data["Label"] = test_data["Color"].map(label_mapping)

X_train = train_data[["X1", "X2"]].values
y_train = train_data["Label"].values
X_test = test_data[["X1", "X2"]].values
y_test = test_data["Label"].values

C = 1.0
model = SVC(kernel="linear", C=C)
model.fit(X_train, y_train)

print(f"Количество опорных векторов: {len(model.support_)}")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axes[0], cmap=plt.cm.Greens)
axes[0].set_title("Confusion Matrix - Train")
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[1], cmap=plt.cm.Greens)
axes[1].set_title("Confusion Matrix - Test")
plt.show()

cmap_light = ListedColormap(['#FF0000', '#228B22'])
cmap_points = ListedColormap(['red', 'green'])

fig, ax = plt.subplots(figsize=(8, 6))
disp = DecisionBoundaryDisplay.from_estimator(
    model,
    X_train,
    response_method="predict",
    cmap=cmap_light,
    alpha=0.3,
    ax=ax,
    xlabel="X1",
    ylabel="X2"
)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_points, s=50, edgecolors="k")
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
           s=100, facecolors='none', edgecolors='k', linewidths=1.8)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Class 0 (red)', markerfacecolor='red', markersize=8, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Class 1 (green)', markerfacecolor='green', markersize=8, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Support Vectors', markerfacecolor='none', markeredgecolor='k', markersize=10, linewidth=1.5)
]
ax.legend(handles=legend_elements)
ax.set_title("SVC (Linear kernel) - Train")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
disp_test = DecisionBoundaryDisplay.from_estimator(
    model,
    X_test,
    response_method="predict",
    cmap=cmap_light,
    alpha=0.3,
    ax=ax,
    xlabel="X1",
    ylabel="X2"
)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_points, s=50, edgecolors="k")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Class 0 (red)', markerfacecolor='red', markersize=8, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Class 1 (green)', markerfacecolor='green', markersize=8, markeredgecolor='k')
]
ax.legend(handles=legend_elements)
ax.set_title("SVC (Linear kernel) - Test")
plt.show()
