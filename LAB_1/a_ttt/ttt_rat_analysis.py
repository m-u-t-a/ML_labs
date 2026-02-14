import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

column_names = ['top-left', 'top-middle', 'top-right',
                'middle-left', 'middle-middle', 'middle-right',
                'bottom-left', 'bottom-middle', 'bottom-right',
                'class']

data_frame = pd.read_csv(
    'C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/datasets/tic_tac_toe.txt',
    names=column_names
)

X = data_frame.iloc[:, :-1]
y = (data_frame['class'] == 'positive').astype(int)

encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

model = CategoricalNB()

test_sizes = np.linspace(0.05, 0.95, 10)
train_scores = []
test_scores = []

for ts in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=ts, stratify=y, random_state=666
    )

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    train_scores.append(train_acc)
    test_scores.append(test_acc)

plt.figure(figsize=(8, 6))
plt.plot(test_sizes, train_scores, marker='o', color='grey', label='Train accuracy')
plt.plot(test_sizes, test_scores, marker='o', color='green', label='Test accuracy')

for x, y in zip(test_sizes, train_scores):
    plt.annotate(f'{y:.3f}',(x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                fontsize=9, fontweight='bold', color='grey')

for x, y in zip(test_sizes, test_scores):
    plt.annotate(f'{y:.3f}',(x, y), textcoords="offset points", xytext=(0, -15), ha='center',
                fontsize=9, fontweight='bold', color='green')

plt.xlabel('Test dataset ratio')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.grid(True)
plt.legend()
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.show()