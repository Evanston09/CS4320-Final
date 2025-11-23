# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils import train, validate
import os
from datetime import datetime


def main():
    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    iterations_list = list(range(100000, 1000001, 1000))
    folds = list(range(5))

    output_file = 'data/experiments.csv'

    if not os.path.exists(output_file):
        df_init = pd.DataFrame(columns=[
            'fold', 'alpha', 'iterations', 'speaker', 'final_train_J',
            'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f1'
        ])
        df_init.to_csv(output_file, index=False)

    total_combinations = len(alphas) * len(iterations_list) * len(folds)
    current = 0

    print(f"Total combinations: {total_combinations}")
    print(f"Alphas: {alphas}")
    print(f"Iterations: {len(iterations_list)} values from {min(iterations_list)} to {max(iterations_list)}")
    print(f"Folds: {folds}")
    print(f"Results will be saved to: {output_file}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for fold in folds:
        train_df = pd.read_csv(f"data/folds/train{fold}.csv")
        val_df_original = pd.read_csv(f"data/folds/val{fold}.csv")

        for alpha in alphas:
            for iterations in iterations_list:
                current += 1

                print(f"[{current}/{total_combinations}] Fold {fold}, Alpha {alpha}, Iterations {iterations}")

                weights, js = train(train_df, "speaker", iterations, alpha)

                val_df = val_df_original.copy()
                cms = validate(val_df, weights, "speaker")

                for speaker, cm in cms.items():
                    row = {
                        'fold': fold,
                        'alpha': alpha,
                        'iterations': iterations,
                        'speaker': speaker,
                        'final_j': js[speaker][0][0],
                        **cm
                    }

                    df_row = pd.DataFrame([row])
                    df_row.to_csv(output_file, mode='a', header=False, index=False)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
