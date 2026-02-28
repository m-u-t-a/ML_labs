import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/spam7.csv")

print("Размер:", data.shape)
print("\nРаспределение классов:")
print(data["yesno"].value_counts())

X = data.drop(columns=["yesno"])
y = (data["yesno"] == "y").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=666, stratify=y
)

# base tree
tree = DecisionTreeClassifier(criterion="gini", random_state=666)
tree.fit(X_train, y_train)

print(f"\nБазовое дерево:")
print(f"  Depth:     {tree.get_depth()}")
print(f"  Leaves:    {tree.get_n_leaves()}")
print(f"  Train acc: {accuracy_score(y_train, tree.predict(X_train)):.4f}")
print(f"  Test acc:  {accuracy_score(y_test, tree.predict(X_test)):.4f}")

# Acc vs Depth
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

best_depth = list(depths)[test_accs.index(max(test_accs))]
print(f"\nBest depth: {best_depth}, Test acc: {max(test_accs):.4f}")

# Acc vs min_samples_leaf
min_leaves = range(1, 50)
train_accs_leaf, test_accs_leaf = [], []

for ml in min_leaves:
    m = DecisionTreeClassifier(criterion="gini", max_depth=best_depth,
                                min_samples_leaf=ml, random_state=666)
    m.fit(X_train, y_train)
    train_accs_leaf.append(accuracy_score(y_train, m.predict(X_train)))
    test_accs_leaf.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(min_leaves, train_accs_leaf, label="Train", marker="o")
plt.plot(min_leaves, test_accs_leaf, label="Test", marker="o")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title(f"Accuracy vs min_samples_leaf (max_depth={best_depth})")
plt.legend()
plt.grid(True)
plt.show()

best_leaf = list(min_leaves)[test_accs_leaf.index(max(test_accs_leaf))]
print(f"Best min_samples_leaf: {best_leaf}, Test acc: {max(test_accs_leaf):.4f}")

# splitting criterion
print("\nSplitting criterion")
print(f"{'Criterion':>10} | {'Train acc':>10} | {'Test acc':>10}")
print("-" * 38)

for criterion in ["gini", "entropy", "log_loss"]:
    m = DecisionTreeClassifier(criterion=criterion, max_depth=best_depth,
                                min_samples_leaf=best_leaf, random_state=666)
    m.fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, m.predict(X_test))
    print(f"{criterion:>10} | {tr:>10.4f} | {te:>10.4f}")

# final results
best_tree = DecisionTreeClassifier(criterion="gini", max_depth=best_depth,
                                 random_state=666)
best_tree.fit(X_train, y_train)

print(f"\nFinal tree:")
print(f"  Train acc: {accuracy_score(y_train, best_tree.predict(X_train)):.4f}")
print(f"  Test acc:  {accuracy_score(y_test, best_tree.predict(X_test)):.4f}")

cv_scores = cross_val_score(best_tree, X, y, cv=5, scoring="accuracy")
print(f"  CV mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\nClassification report:")
print(classification_report(y_test, best_tree.predict(X_test),
                             target_names=["not spam", "spam"]))

# fetures importances
importances = pd.Series(best_tree.feature_importances_, index=X.columns)
importances = importances[importances > 0].sort_values(ascending=True)

plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="steelblue")
plt.title("Feature importances")
plt.xlabel("Importance")
plt.grid(True, axis="x")
plt.tight_layout()
plt.show()

# final tree visual
plt.figure(figsize=(20, 8))
plot_tree(best_tree, feature_names=X.columns,
          class_names=["not spam", "spam"],
          filled=True, rounded=True, fontsize=9)
plt.title(f"Optimal tree (depth={best_depth}, min_leaf={best_leaf})")
plt.tight_layout()
plt.show()
