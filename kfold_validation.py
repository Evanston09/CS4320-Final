# -*- coding: utf-8 -*-
"""
General Logistic Regression

"""

import pandas as pd
import numpy as np
from utils import train, validate


def train_on_everything():
    df = pd.read_csv("data/data.csv")
    yCol = "speaker"

    # TRAIN THE SYSTEM  (20000, 0.0000006)
    iterations = 200000
    alpha = 0.1

    weights_dict, _ = train(df, yCol, iterations, alpha)

    for speaker in weights_dict:
        np.save(f"weights/{speaker}.npy", weights_dict[speaker])


def do_k_fold(k_num=5):
    for k in range(k_num):
        train_df = pd.read_csv(f"data/folds/train{k}.csv")
        val_df = pd.read_csv(f"data/folds/val{k}.csv")
        yCol = "speaker"

        # TRAIN THE SYSTEM  (20000, 0.0000006)
        iterations = 100000
        alpha = 0.1

        weights_dict, _ = train(train_df, yCol, iterations, alpha)

        cm = validate(val_df, weights_dict, yCol)
        print(cm)


if __name__ == "__main__":
    do_k_fold()
