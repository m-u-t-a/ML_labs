import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, train_test_split
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

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=666
)

encoder = OrdinalEncoder()
X_train = encoder.fit_transform(X_train_df)
X_test = encoder.transform(X_test_df)

model = CategoricalNB()

train_sizes = np.linspace(0.05, 1.0, 10)

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
print(f"Final test accuracy (full train): {final_test_accuracy:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(train_sizes_abs, train_scores_mean, marker='o', color='grey', label='Train accuracy')
plt.plot(train_sizes_abs, val_scores_mean, marker='o', color='green', label='Cross-validation accuracy')

for x, y in zip(train_sizes_abs, train_scores_mean):
    plt.annotate(f'{y:.3f}',(x, y), textcoords="offset points", xytext=(0, 10), ha='center',
                fontsize=9, fontweight='bold', color='grey')

for x, y in zip(train_sizes_abs, val_scores_mean):
    plt.annotate(f'{y:.3f}',(x, y), textcoords="offset points", xytext=(0, -15), ha='center',
                fontsize=9, fontweight='bold', color='green')

plt.xlabel('Train dataset volume')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.grid(True)
plt.legend()
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.show()