import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calc_stats(tp, tn, fp, fn):
    accuracy = (tp + tn)/(tn+tp+fn+fp)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = (2/((1/precision)+(1/recall)))
    return accuracy, precision, recall, f1


# Logistic Regression Hypothesis Function
def hwX(w, X):
    return 1 / (1 + np.exp(-X @ w))


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


def train(df, yCol, iter, alpha):
    X, y_dict, speakers = prepare_one_vs_all(df, yCol)
    print("Speakers:", speakers)

    weights_dict = {}
    final_j = {}
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for idx, speaker in enumerate(speakers):
        print(df.columns)
        m = len(df.index)
        n = len(df.columns)
        print("n=", n, "m=", m)

        print(f"\nTraining model for speaker: {speaker}")
        y = y_dict[speaker]
        w = np.zeros((n, 1))
        print("Initial Error (J)=", J(w, X, y, m))

        errors_to_plot = []
        iterations_to_plot = []

        for k in range(iter):
            if k % 50000 == 0:
                print(k)
            error = J(w, X, y, m)
            if k >= iter - 10000:
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
        final_j[speaker] = final_error
        print("Final J=", final_error)

    print("\nTraining complete for all speakers")

    plt.xlabel("Iteration")
    plt.ylabel("Cost (J)")
    plt.title("Training Cost Over Time")
    plt.legend()
    plt.savefig(f'data/graphs/{iter}_{alpha}.png')

    return weights_dict, final_j


def validate(val_df, weights_dict, yCol):
    y_vals = val_df.pop(yCol).to_numpy()
    X_vals = val_df.to_numpy()

    X_vals = np.insert(X_vals, 0, 1, axis=1)

    confusion_matrix = {speaker: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for speaker in weights_dict}

    for idx, row in enumerate(X_vals):
        predictions = {}
        for speaker, weights in weights_dict.items():
            prob = hwX(weights, row)
            predictions[speaker] = prob

        predicted_speaker = max(predictions, key=predictions.get)
        actual_speaker = y_vals[idx]

        print(f"Predicted: {predicted_speaker}, Actual: {actual_speaker}")

        for speaker in weights_dict:
            if predicted_speaker == speaker and actual_speaker == speaker:
                confusion_matrix[speaker]['tp'] += 1
            elif predicted_speaker != speaker and actual_speaker != speaker:
                confusion_matrix[speaker]['tn'] += 1
            elif predicted_speaker == speaker and actual_speaker != speaker:
                confusion_matrix[speaker]['fp'] += 1
            elif predicted_speaker != speaker and actual_speaker == speaker:
                confusion_matrix[speaker]['fn'] += 1

    for speaker in weights_dict:
        cm = confusion_matrix[speaker]
        tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']

        accuracy, precision, recall, f1 = calc_stats(tp, tn, fp, fn)

        confusion_matrix[speaker]['accuracy'] = accuracy
        confusion_matrix[speaker]['precision'] = precision
        confusion_matrix[speaker]['recall'] = recall
        confusion_matrix[speaker]['f1'] = f1

    return confusion_matrix

