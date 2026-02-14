import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data_frame = pd.read_csv(
    'C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/datasets/spam.csv', index_col=0
)

X = data_frame.iloc[:, :-1].values
y = (data_frame['type'] == 'spam').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=666
)

model = GaussianNB()

train_sizes = np.linspace(0.001, 1.0, 10)

train_sizes_abs, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    shuffle=True,
    random_state=666
)

train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

model.fit(X_train, y_train)
final_test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Final test accuracy: {final_test_accuracy:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(train_sizes_abs, train_scores_mean, marker='o', color='grey', label='Train accuracy')
plt.plot(train_sizes_abs, val_scores_mean, marker='o', color='orange', label='Cross-validation accuracy')

for x, y in zip(train_sizes_abs, train_scores_mean):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                 fontsize=9, fontweight='bold', color='grey')

for x, y in zip(train_sizes_abs, val_scores_mean):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, -15), ha='center',
                 fontsize=9, fontweight='bold', color='orange')

plt.xlabel('Train dataset volume')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.grid(True)
plt.legend()
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.show()
