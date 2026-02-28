import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

train_data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/svmdata_b.txt", sep="\t")
test_data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/svmdata_b_test.txt", sep="\t")

label_mapping = {label: idx for idx, label in enumerate(train_data["Colors"].unique())}
train_data["Label"] = train_data["Colors"].map(label_mapping)
test_data["Label"] = test_data["Colors"].map(label_mapping)

X_train = train_data[["X1", "X2"]].values
y_train = train_data["Label"].values

C = 1.0
model = SVC(kernel="linear", C=C)
model.fit(X_train, y_train)

print(f"Количество опорных векторов: {len(model.support_)}")

cmap_light = ListedColormap(['#FF0000', '#0000CD'])
cmap_points = ListedColormap(['red', 'blue'])

fig, ax = plt.subplots(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(
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
    Line2D([0], [0], marker='o', color='w', label='Class 1 (blue)', markerfacecolor='blue', markersize=8, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Support Vectors', markerfacecolor='none', markeredgecolor='k', markersize=10, linewidth=1.5)
]
ax.legend(handles=legend_elements)
ax.set_title("SVC (Linear kernel) - Train")
plt.show()
