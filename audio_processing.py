from pydub import AudioSegment, silence
import numpy as np
import os
import librosa
import csv
from pathlib import Path

from pydub.utils import make_chunks


def clean_audio(file_dir, output_dir, format="m4a", segment_length_ms=15000):
    audio = AudioSegment.from_file(file_dir, format=format)

    non_silent = silence.split_on_silence(
        audio,
        silence_thresh=audio.dBFS - 16
    )

    non_silent_combined = AudioSegment.empty()
    for chunk in non_silent:
        non_silent_combined += chunk

    chunked = make_chunks(non_silent_combined, segment_length_ms)

    # This works ts just confused
    for i, chunk in enumerate(chunked):
        base = os.path.splitext(os.path.basename(file_dir))[0]
        chunk.export(os.path.join(output_dir, base + str(i) + ".mp3"))


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

    f0, voiced_flags, _ = librosa.pyin(y=y, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz('C7')))

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


def get_feature_names(n_mfcc=20):
    names = []
    for i in range(n_mfcc):
        names.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std',
                     f'mfcc_{i}_min', f'mfcc_{i}_max'])
    for i in range(n_mfcc):
        names.extend([f'delta_mfcc_{i}_mean', f'delta_mfcc_{i}_std',
                     f'delta_mfcc_{i}_min', f'delta_mfcc_{i}_max'])
    names.extend(['f0_mean', 'f0_std', 'f0_min', 'f0_max'])
    names.extend(['centroid_mean', 'centroid_std'])
    names.extend(['rolloff_mean', 'rolloff_std'])
    names.append('speaker')

    return names


def process_dir(base_dir):
    directories = os.listdir(path=base_dir)

    data = []

    for directory in directories:
        person_dir = os.path.join(base_dir, directory)

        if not os.path.isdir(person_dir):
            continue

        speaker_name = directory

        raw_dir = os.path.join(person_dir, 'raw')
        for entry in os.listdir(raw_dir):
            clean_audio(os.path.join(raw_dir, entry), os.path.join(person_dir, 'processed'))

        processed_dir = os.path.join(person_dir, 'processed')
        for entry in os.listdir(processed_dir):
            features = extract_features(os.path.join(processed_dir, entry))
            features.append(speaker_name)
            data.append(features)

    data.insert(0, get_feature_names())

    # Saving a list of lists
    with open('data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


if __name__ == "__main__":
    process_dir("data")
    # extract_features("test.mp3")
