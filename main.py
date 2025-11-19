# -*- coding: utf-8 -*-
"""
General Logistic Regression

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Create functions
# Logistic Regression Hypothesis Function
def hwX(w, X):
    return 1 / (1 + np.exp(-X @ w))


# Compute all the individual costs
def Cost(w, X, y):
    H = 1 / (1 + np.exp(-X @ w))
    return -(y * np.log(H)) - (1 - y) * np.log(1 - H)


# Average cost over all training data
def J(w, X, y, m):
    H = (1/(1+np.exp(-X @ w)))
    eps = 1e-10
    H = np.clip(H, eps, 1 - eps)
    cost = -(y * np.log(H)) - (1-y) * np.log(1-H)
    xm1 = np.ones((1, m))
    return (1/m) * (xm1 @ cost)


# Gradient Descent
def GD(w, X, y, m, alpha):
    H = 1 / (1 + np.exp(-X @ w))

    return w - ((alpha / m) * (H - y).T @ X).T


# Function to create X and y matrices
def array_maker(df, yCol):
    x_headers = list(df)
    x_headers.remove(yCol)

    data_array = df[x_headers].to_numpy()
    array_of_1s = np.ones([1, len(df)])

    X = np.insert(data_array, 0, array_of_1s, axis=1)
    Y = df[[yCol]].to_numpy()

    return X, Y


def prepare_one_vs_all(df, yCol):
    speakers = df[yCol].unique()
    x_headers = list(df)
    x_headers.remove(yCol)

    data_array = df[x_headers].to_numpy()
    array_of_1s = np.ones([1, len(df)])
    X = np.insert(data_array, 0, array_of_1s, axis=1)

    y_dict = {}
    for speaker in speakers:
        y_dict[speaker] = (df[[yCol]] == speaker).astype(int).to_numpy()

    return X, y_dict, speakers


def get_mean_std(df, dropcol):
    tdf = df.drop(dropcol, axis=1)
    means = tdf.mean()
    stds = tdf.std()
    return means, stds


def s_df(df, dropcol, mean_array, std_array):
    tdf = df.drop(dropcol, axis=1)
    tdf = ((tdf - mean_array) / std_array)
    tdf[dropcol] = df[dropcol].values
    return tdf


# MAIN
df = pd.read_csv("data.csv")
yCol = "speaker"

mean_array, std_array = get_mean_std(df, yCol)
s_df = s_df(df, yCol, mean_array, std_array)

print(df.columns)
m = len(df.index)
n = len(df.columns)
print("n=", n, "m=", m)
X, y_dict, speakers = prepare_one_vs_all(s_df, "speaker")
print("Speakers:", speakers)
print(X)

input()
# TRAIN THE SYSTEM  (20000, 0.0000006)
iterations = 2000000
alpha = 0.001

weights_dict = {}

colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx, speaker in enumerate(speakers):
    print(f"\nTraining model for speaker: {speaker}")
    y = y_dict[speaker]
    w = np.zeros((n, 1))
    print("Initial Error (J)=", J(w, X, y, m))

    errors_to_plot = []
    iterations_to_plot = []

    for k in range(iterations):

        if k % 50000 == 0:
            print(k)
        error = J(w, X, y, m)
        if k >= iterations - 100:
            errors_to_plot.append(error[0][0])
            iterations_to_plot.append(k)
        w = GD(w, X, y, m, alpha)

    plt.scatter(iterations_to_plot, errors_to_plot,
                color=colors[idx % len(colors)],
                label=speaker,
                alpha=0.6)

    print("The weights:")
    print(w)
    weights_dict[speaker] = w

    final_error = J(w, X, y, m)
    print("Final J=", final_error)


print("\nTraining complete for all speakers")

for speaker in weights_dict:
    np.save(f"weights/{speaker}.npy", weights_dict[speaker])

plt.xlabel("Iteration")
plt.ylabel("Cost (J)")
plt.title("Training Cost Over Time")
plt.legend()
plt.show()
