import os
import csv

import numpy as np
import pandas as pd
import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


model = load_silero_vad()


def clean_audio(file_dir, output_dir, format="m4a", segment_length_ms=5000):
    wav = read_audio(file_dir)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
    )
    audio = AudioSegment.from_file(file_dir, format=format)

    # Silero VAD always uses 16kHz sample rate
    silero_sr = 16000

    only_speech = AudioSegment.empty()
    for speech in speech_timestamps:
        start = (speech['start']/silero_sr) * 1000
        end = (speech['end']/silero_sr) * 1000

        only_speech += audio[start:end]

    chunked = make_chunks(only_speech, segment_length_ms)

    for i, chunk in enumerate(chunked):
        # Skip chunks shorter than 1 second (1000ms)
        if len(chunk) < 1000:
            continue
        base = os.path.splitext(os.path.basename(file_dir))[0]
        chunk.export(os.path.join(output_dir, f"{base}{i}.mp3"))


# Get MFCC, delta MFCC, Pitch stats, Centroid, rolloff
def extract_features(file_name):
    y, sr = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)

    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)

    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfcc_mean = np.mean(delta_mfccs, axis=1)
    delta_mfcc_std = np.std(delta_mfccs, axis=1)
    delta_mfcc_min = np.min(delta_mfccs, axis=1)
    delta_mfcc_max = np.max(delta_mfccs, axis=1)

    f0, voiced_flags, _ = librosa.pyin(
        y=y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7"))
    )

    f0_voiced = f0[voiced_flags]

    f0_mean = np.mean(f0_voiced)
    f0_std = np.std(f0_voiced)
    f0_min = np.min(f0_voiced)
    f0_max = np.max(f0_voiced)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    features = []
    features.extend(mfcc_mean)
    features.extend(mfcc_std)
    features.extend(mfcc_min)
    features.extend(mfcc_max)
    features.extend(delta_mfcc_mean)
    features.extend(delta_mfcc_std)
    features.extend(delta_mfcc_min)
    features.extend(delta_mfcc_max)
    features.extend([f0_mean, f0_std, f0_min, f0_max])
    features.extend([centroid_mean, centroid_std])
    features.extend([rolloff_mean, rolloff_std])

    return features


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


def get_feature_names(n_mfcc=20):
    names = []
    for i in range(n_mfcc):
        names.extend([
            f'mfcc_{i}_mean', f'mfcc_{i}_std',
            f'mfcc_{i}_min', f'mfcc_{i}_max'
        ])
    for i in range(n_mfcc):
        names.extend([
            f'delta_mfcc_{i}_mean', f'delta_mfcc_{i}_std',
            f'delta_mfcc_{i}_min', f'delta_mfcc_{i}_max'
        ])
    names.extend(['f0_mean', 'f0_std', 'f0_min', 'f0_max'])
    names.extend(['centroid_mean', 'centroid_std'])
    names.extend(['rolloff_mean', 'rolloff_std'])
    names.append('speaker')

    return names


def generate_stratified_k_folds(df, ouput_col, random_state=42, k=5):
    speakers = {}
    np.random.seed(42)

    for idx, speaker in enumerate(df[ouput_col]):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(idx)

    for speaker in speakers:
        np.random.shuffle(speakers[speaker])

    speaker_folds = {}
    for speaker, indices in speakers.items():
        speaker_folds[speaker] = np.array_split(indices, k)

    for fold_k in range(k):
        val_indexes = []
        train_indexes = []
        for folds in speaker_folds.values():
            val_indexes.extend(folds[fold_k])

            for train_fold_k in range(k):
                if train_fold_k != fold_k:
                    train_indexes.extend(folds[train_fold_k])

        train_df = df.iloc[train_indexes]
        val_df = df.iloc[val_indexes]

        train_df.to_csv(f"data/folds/train{fold_k}.csv", index=False)
        val_df.to_csv(f"data/folds/val{fold_k}.csv", index=False)


def process_dir(base_dir):
    directories = os.listdir(path=base_dir)

    data = []

    for directory in directories:
        print("Processing directory", directory)
        person_dir = os.path.join(base_dir, directory)

        if not os.path.isdir(person_dir):
            continue

        speaker_name = directory

        raw_dir = os.path.join(person_dir, 'raw')
        processed_dir = os.path.join(person_dir, 'processed')

        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        for entry in os.listdir(raw_dir):
            print("Processing entry", entry)
            clean_audio(os.path.join(raw_dir, entry), processed_dir)

        for entry in os.listdir(processed_dir):
            features = extract_features(os.path.join(processed_dir, entry))
            features.append(speaker_name)
            data.append(features)

    df = pd.DataFrame(data, columns=get_feature_names())

    means, stds = get_mean_std(df, 'speaker')
    df_standardized = s_df(df, 'speaker', means, stds)

    np.savez('data/normalization_params.npz', means=means.values, stds=stds.values)
    print("Saved normalization parameters to data/normalization_params.npz")

    df_standardized.to_csv('data/data.csv', index=False)


if __name__ == "__main__":
    process_dir("data/speakers")
    df = pd.read_csv("data/data.csv")
    generate_stratified_k_folds(df, "speaker")
