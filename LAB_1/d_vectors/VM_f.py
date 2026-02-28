import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

train_data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/svmdata_e.txt", sep="\t")
test_data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/svmdata_e_test.txt", sep="\t")

label_mapping = {label: idx for idx, label in enumerate(train_data["Colors"].unique())}
train_data["Label"] = train_data["Colors"].map(label_mapping)
test_data["Label"] = test_data["Colors"].map(label_mapping)

X_train = train_data[["X1", "X2"]].values
y_train = train_data["Label"].values
X_test = test_data[["X1", "X2"]].values
y_test = test_data["Label"].values

cmap_light = ListedColormap(['#FF6666', '#66CC66'])
cmap_points = ListedColormap(['red', 'green'])

kernels = [
    ("poly deg=1",     {"kernel": "poly",    "degree": 1, "C": 10}),
    ("poly deg=2",     {"kernel": "poly",    "degree": 2, "C": 10}),
    ("poly deg=5",     {"kernel": "poly",    "degree": 5, "C": 10}),
    ("sigmoid",        {"kernel": "sigmoid", "C": 10}),
    ("poly deg=3",     {"kernel": "poly",    "degree": 3, "C": 10}),
    ("poly deg=4",     {"kernel": "poly",    "degree": 4, "C": 10}),
    ("RBF (gaussian)", {"kernel": "rbf",     "C": 10})
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Different kernels (gamma=default)", fontsize=14)
axes = axes.flatten()

print("=== Different kernels (gamma=default) ===")
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
    ax.set_title(f"{name} | Train: {train_acc:.2f} | Test: {test_acc:.2f}")

axes[-1].set_visible(False)
plt.tight_layout()
plt.show()

# Gamma changing
gamma_values = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("RBF kernel — эффект гаммы (оверфиттинг)", fontsize=14)
axes = axes.flatten()

print("\n=== RBF, different gamma ===")
print(f"{'Gamma':>10} | {'Train acc':>10} | {'Test acc':>10} | {'Num SV':>8}")
print("-" * 47)

for i, gamma in enumerate(gamma_values):
    model = SVC(kernel="rbf", C=10, gamma=gamma)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{gamma:>10} | {train_acc:>10.4f} | {test_acc:>10.4f} | {len(model.support_):>8}")

    ax = axes[i]
    DecisionBoundaryDisplay.from_estimator(
        model, X_train, response_method="predict",
        cmap=cmap_light, alpha=0.3, ax=ax, xlabel="X1", ylabel="X2"
    )
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_points, s=50, edgecolors="k")
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='k', linewidths=1.8)
    ax.set_title(f"gamma={gamma} | Train: {train_acc:.2f} | Test: {test_acc:.2f}")

plt.tight_layout()
plt.show()
