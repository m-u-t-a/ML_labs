import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

from LAB_1.b_dots_NB.process_dots import generate_class, draw_data, save_dataset

np.random.seed(666)

mean_neg = np.array([18, 18])
std_neg = np.array([4, 4])
n_neg = 10

mean_pos = np.array([18, 18])
std_pos = np.array([2, 2])
n_pos = 90

X_neg, y_neg = generate_class(mean_neg, std_neg, n_neg, label=-1)
X_pos, y_pos = generate_class(mean_pos, std_pos, n_pos, label=1)

X = np.vstack([X_neg, X_pos])
y = np.hstack([y_neg, y_pos])

# draw_data(X_neg, X_pos)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=666
)

model = GaussianNB()
model.fit(X_train, y_train)

# threshold = 0.95
#
# y_scores = model.predict_proba(X_train)[:, 1]
# y_pred_train = np.where(y_scores >= threshold, 1, -1)
#
# y_scores = model.predict_proba(X_test)[:, 1]
# y_pred_test = np.where(y_scores >= threshold, 1, -1)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy:  {test_accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred_test)

print("\nConfusion Matrix:")
print(cm)

TN, FP, FN, TP = cm.ravel()

print(f"\nTrue Negative (TN):  {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")
print(f"True Positive (TP):  {TP}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["\" -1 \"", "\" 1 \""])

disp.plot(cmap="Greens")
plt.title("Confusion Matrix")
plt.show()

# y_proba = model.predict_proba(X_test)[:, 1]
#
# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
#
# roc_auc = roc_auc_score(y_test, y_proba)

# print(f"\nROC AUC: {roc_auc:.4f}")
#
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', color = 'red')
# plt.plot([0, 1], [0, 1], label='Random Guess',  linestyle='-.', color = 'blue')
#
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.grid(True)
# plt.show()

# y_scores = model.predict_proba(X)[:, 1]
#
# precision, recall, thresholds = precision_recall_curve(y, y_scores)
#
# avg_precision = average_precision_score(y, y_scores)
# print(f"Average Precision (PR AUC): {avg_precision:.3f}")
#
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, color='orange', marker='o', label=f'PR curve (PR AUC={avg_precision:.3f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.grid(True)
# plt.legend()
# plt.ylim(0.5, 1.05)
# plt.xlim(0.5, 1.05)
# plt.tight_layout()
# plt.show()