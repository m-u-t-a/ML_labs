import numpy as np
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

draw_data(X_neg, X_pos)

save_dataset(X, y, "C:/Users/Sasha/PycharmProjects/ML_labs/LAB_1/z_datasets/dots_dataset.txt")
