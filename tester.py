#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: evank
"""
import numpy as np
import pandas as pd
from audio_processing import extract_features


# Logistic Regression Hypothesis Function
def hwX(w, X):
    return 1 / (1 + np.exp(-X @ w))


def main():
    yCol = "speaker"
    training_df = pd.read_csv("data/data.csv")

    # Load normalization parameters from saved file
    norm_params = np.load("data/normalization_params.npz")
    means = norm_params['means']
    stds = norm_params['stds']

    speakers = training_df["speaker"].unique()
    print(f"Available speakers: {speakers}")

    weights_dict = {}
    for speaker in speakers:
        try:
            weights_dict[speaker] = np.load(f"weights/{speaker}.npy")
            print(f"Loaded weights for speaker: {speaker}")
        except FileNotFoundError:
            print(f"Warning: Weight file {speaker}.npy not found")

    if not weights_dict:
        print("Error: No weight files found. Please train the model first.")
        return

    audio_file = input("Enter the path to the audio file: ")

    print(f"\nExtracting features from: {audio_file}")
    try:
        features = extract_features(audio_file)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return

    normalized_features = ((features - means) / stds)
    normalized_features = np.array(normalized_features).reshape(1, -1)
    array_of_1s = np.ones([1, 1])

    normalized_features = np.insert(normalized_features,
                                    0,
                                    array_of_1s,
                                    axis=1
                                    )

    print("\nPredictions")
    predictions = {}

    for speaker, weights in weights_dict.items():
        prob = hwX(weights, normalized_features)[0][0]
        predictions[speaker] = prob
        print(f"{speaker}: {prob:.4f} ({prob*100:.2f}%)")

    print(predictions)
    predicted_speaker = max(predictions, key=predictions.get)
    confidence = predictions[predicted_speaker]

    print("\nResult")
    print(f"Predicted speaker: {predicted_speaker}")
    print(f"Confidence: {confidence*100:.2f}%")

    cutoff = 0.5
    print(f"\nBinary predictions (cutoff = {cutoff}):")
    for speaker, prob in predictions.items():
        binary = "YES" if prob > cutoff else "NO"
        print(f"  {speaker}: {binary}")


if __name__ == "__main__":
    main()
