import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

train_data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/svmdata_c.txt", sep="\t")
test_data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/svmdata_c_test.txt", sep="\t")

label_mapping = {label: idx for idx, label in enumerate(train_data["Colors"].unique())}
train_data["Label"] = train_data["Colors"].map(label_mapping)
test_data["Label"] = test_data["Colors"].map(label_mapping)

X_train = train_data[["X1", "X2"]].values
y_train = train_data["Label"].values
X_test = test_data[["X1", "X2"]].values
y_test = test_data["Label"].values

kernels = [
    ("linear",      {"kernel": "linear",  "C": 10}),
    ("poly deg=1",  {"kernel": "poly",    "degree": 1, "C": 10}),
    ("poly deg=4",  {"kernel": "poly",    "degree": 4, "C": 10}),
    ("poly deg=5",  {"kernel": "poly",    "degree": 5, "C": 10}),
    ("poly deg=2",  {"kernel": "poly",    "degree": 2, "C": 10}),
    ("poly deg=3",  {"kernel": "poly",    "degree": 3, "C": 10}),
    ("sigmoid",     {"kernel": "sigmoid", "C": 10}),
    ("RBF (gaussian)", {"kernel": "rbf",  "C": 10})
]

cmap_light = ListedColormap(['#FF6666', '#66CC66'])
cmap_points = ListedColormap(['red', 'green'])

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

print(f"{'Kernel':>15} | {'Train acc':>10} | {'Test acc':>10} | {'Num SV':>8}")
print("-" * 55)

for i, (name, params) in enumerate(kernels):
    model = SVC(**params)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:>15} | {train_acc:>10.4f} | {test_acc:>10.4f} | {len(model.support_):>8}")

    ax = axes[i]
    DecisionBoundaryDisplay.from_estimator(
        model, X_train, response_method="predict",
        cmap=cmap_light, alpha=0.3, ax=ax, xlabel="X1", ylabel="X2"
    )
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_points, s=50, edgecolors="k")
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='k', linewidths=1.8)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Class 0 (red)', markerfacecolor='red', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Class 1 (green)', markerfacecolor='green', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Support Vectors', markerfacecolor='none', markeredgecolor='k', markersize=10)
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_title(f"{name} | Train: {train_acc:.2f} | Test: {test_acc:.2f}")

plt.tight_layout()
plt.show()
