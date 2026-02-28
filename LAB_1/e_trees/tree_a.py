import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/glass.csv")
data = data.drop(columns=["Id"])

X = data.drop(columns=["Type"])
y = data["Type"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=666, stratify=y
)

# base tree
tree = DecisionTreeClassifier(criterion="gini", random_state=666)
tree.fit(X_train, y_train)

train_acc = accuracy_score(y_train, tree.predict(X_train))
test_acc = accuracy_score(y_test, tree.predict(X_test))
print(f"Base tree:")
print(f"  Depth: {tree.get_depth()}")
print(f"  Leaves: {tree.get_n_leaves()}")
print(f"  Train acc: {train_acc:.4f}")
print(f"  Test acc:  {test_acc:.4f}\n")

# Визуализация полного дерева
plt.figure(figsize=(40, 10))
plot_tree(tree, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True, fontsize=7)
plt.title("Decision tree")
plt.tight_layout()
plt.show()

# acc vs depth
depths = range(1, 20)
train_accs, test_accs = [], []

for d in depths:
    m = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=666)
    m.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, m.predict(X_train)))
    test_accs.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(depths, train_accs, label="Train", marker="o")
plt.plot(depths, test_accs, label="Test", marker="o")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Depth")
plt.legend()
plt.grid(True)
plt.show()

best_depth = depths[test_accs.index(max(test_accs))]
print(f"Best depth by test: {best_depth}, Test acc: {max(test_accs):.4f}")

# splitting criterion
print("\n=== Splitting criterion ===")
print(f"{'Criterion':>10} | {'Max depth':>10} | {'Train acc':>10} | {'Test acc':>10}")
print("-" * 50)

for criterion in ["gini", "entropy", "log_loss"]:
    for max_depth in [None, best_depth]:
        m = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=666)
        m.fit(X_train, y_train)
        tr = accuracy_score(y_train, m.predict(X_train))
        te = accuracy_score(y_test, m.predict(X_test))
        depth_label = str(max_depth) if max_depth else "None"
        print(f"{criterion:>10} | {depth_label:>10} | {tr:>10.4f} | {te:>10.4f}")

# min_samples_leaf
min_leaves = range(1, 20)
train_accs_leaf, test_accs_leaf = [], []

for ml in min_leaves:
    m = DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=ml, random_state=666)
    m.fit(X_train, y_train)
    train_accs_leaf.append(accuracy_score(y_train, m.predict(X_train)))
    test_accs_leaf.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(min_leaves, train_accs_leaf, label="Train", marker="o")
plt.plot(min_leaves, test_accs_leaf, label="Test", marker="o")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title("Accuracy vs min_samples_leaf (max_depth=6)")
plt.legend()
plt.grid(True)
plt.show()

best_leaf = list(min_leaves)[test_accs_leaf.index(max(test_accs_leaf))]
print(f"Best min_samples_leaf: {best_leaf}, Test acc: {max(test_accs_leaf):.4f}")

# max_features
max_features_options = [None, "sqrt", "log2", 2, 3, 4, 5, 6, 7, 8, 9]
train_accs_feat, test_accs_feat = [], []

print("\n=== max_features ===")
print(f"{'max_features':>15} | {'Train acc':>10} | {'Test acc':>10}")
print("-" * 42)

for mf in max_features_options:
    m = DecisionTreeClassifier(criterion="gini", max_depth=6, max_features=mf, random_state=666)
    m.fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, m.predict(X_test))
    train_accs_feat.append(tr)
    test_accs_feat.append(te)
    print(f"{str(mf):>15} | {tr:>10.4f} | {te:>10.4f}")

# final building

best_tree = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=666, min_samples_leaf=5, max_features=5)
best_tree.fit(X_train, y_train)
print("\nFinal tree score:")
print(f"  Train acc: {accuracy_score(y_train, best_tree.predict(X_train)):.4f}")
print(f"  Test acc:  {accuracy_score(y_test, best_tree.predict(X_test)):.4f}")

plt.figure(figsize=(20, 8))
plot_tree(best_tree, feature_names=X.columns,
          class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True, fontsize=9)
plt.title(f"Optimal tree (max_depth=6)")
plt.tight_layout()
plt.show()
