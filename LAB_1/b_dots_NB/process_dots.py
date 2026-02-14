import numpy as np
import matplotlib.pyplot as plt

def generate_class(mean, std, n, label):
    X_raw = np.random.randn(n, 2)

    X_scaled = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)

    X = X_scaled * std + mean

    y = label * np.ones(n)

    return X, y

def draw_data(X_neg, X_pos):
    plt.figure(figsize=(8, 6))

    plt.scatter(X_neg[:, 0], X_neg[:, 1],
                color='red', label='Class -1',
                alpha=0.7, edgecolor='k')

    plt.scatter(X_pos[:, 0], X_pos[:, 1],
                color='blue', label='Class 1',
                alpha=0.7, edgecolor='k')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Dots distribution')
    plt.grid(True)
    plt.legend()
    plt.show()

def save_dataset(X, y, filename):
    data = np.column_stack((X, y))
    np.savetxt(filename, data, fmt="%.5f", header="X1 X2 class", comments='')
