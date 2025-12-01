import wave
from clean_processed_data import reset_processed, reset_speakers
import numpy as np
import os
import sys
import pandas as pd
import threading
from audio_processing import process_dir
import numpy as np
import torch
torch.set_num_threads(1)
import pyaudio
from utils import train, evaluate
import json


ITER = 100000
ALPHA = 0.01

SPEECH = """When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.
The rainbow is a division of white light into many beautiful colors.
These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon.
There is, according to legend, a boiling pot of gold at one end.
People look, but no one ever finds it.
When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.
Throughout the centuries people have explained the rainbow in various ways.
Some have accepted it as a miracle without physical explanation.
To the Hebrews it was a token that there would be no more universal floods.
The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.
The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky.
Others have tried to explain the phenomenon physically.
Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.
Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows.
Many complicated ideas about the rainbow have been formed.
The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases.
The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.
If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow.
This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue."""


def get_speakers():
    if os.path.exists("data/speakers"):
        directories = os.listdir(path="data/speakers")
        speakers = [
            dir
            for dir in directories
            if os.path.isdir(os.path.join("data/speakers", dir))
        ]
    else:
        speakers = []

    return speakers


def load_accuracy_stats():
    stats_file = "data/accuracy_stats.json"
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            return json.load(f)
    return {"correct": 0, "total": 0}


def save_accuracy_stats(stats):
    stats_file = "data/accuracy_stats.json"
    os.makedirs("data", exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)


def display_accuracy_stats():
    stats = load_accuracy_stats()
    if stats["total"] == 0:
        print("No predictions recorded yet.")
    else:
        accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"Accuracy: {stats['correct']}/{stats['total']} ({accuracy:.2f}%)")


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def record_audio(filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 if sys.platform == "darwin" else 2
    RATE = 44100

    with wave.open(filename, "wb") as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

        print("Recording...")

        continue_recording = True

        def stop():
            input("Press Enter to stop the recording:")
            nonlocal continue_recording
            continue_recording = False

        stop_listener = threading.Thread(target=stop)
        stop_listener.start()

        while continue_recording:
            wf.writeframes(stream.read(CHUNK))

        print("Done")

        stream.close()
        p.terminate()


def detect_realtime(model, weights_dict):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 16000
    CHUNK = int(SAMPLE_RATE / 10)

    audio = pyaudio.PyAudio()
    num_samples = 512

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    data = []
    confidence = []

    print("Started Recording")
    length = 0

    continue_recording = True
    def stop():
        input("Press Enter to stop the recording:")
        nonlocal continue_recording
        continue_recording = False

    stop_listener = threading.Thread(target=stop)
    stop_listener.start()

    grace = 32
    while continue_recording:
        audio_chunk = stream.read(num_samples, exception_on_overflow=False)

        audio_int16 = np.frombuffer(audio_chunk, np.int16)

        audio_float32 = int2float(audio_int16)

        confidence = model(torch.from_numpy(audio_float32), 16000).item()

        if confidence > .5:
            # 512 samples is equal to .032 seconds
            if grace == 32:
                print("Started talking")
            grace = 0
            length += .032
            data.append(audio_chunk)
        else:
            if (grace < 32):
                grace += 1
            else:
                if grace == 32 and length > 0:
                    print("Stopped talking")
                data.clear()
                length = 0

        if length >= 5:
            with wave.open("temp.wav", "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)

                for chunk in data:
                    wf.writeframes(chunk)

            print("Evaluating...")
            evaluate("temp.wav", weights_dict)

            # Ask for feedback
            feedback = input("Was the prediction correct? (y/n): ").lower().strip()
            stats = load_accuracy_stats()
            stats["total"] += 1
            if feedback == "y":
                stats["correct"] += 1
                print("Recorded as correct.")
            else:
                print("Recorded as incorrect.")
            save_accuracy_stats(stats)

            # Display current accuracy
            accuracy = (stats["correct"] / stats["total"]) * 100
            print(f"Current accuracy: {stats['correct']}/{stats['total']} ({accuracy:.2f}%)")

            # Reset after evaluation to prevent repeated evaluations
            data.clear()
            length = 0
            grace = 32

    print("Stopped the recording")

def main():
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
    )

    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗███████╗██╗  ██╗██████╗ ██████╗  ██████╗             ║
║  ██╔════╝██╔════╝██║  ██║╚════██╗╚════██╗██╔═████╗            ║
║  ██║     ███████╗███████║ █████╔╝ █████╔╝██║██╔██║            ║
║  ██║     ╚════██║╚════██║ ╚═══██╗██╔═══╝ ████╔╝██║            ║
║  ╚██████╗███████║     ██║██████╔╝███████╗╚██████╔╝            ║
║   ╚═════╝╚══════╝     ╚═╝╚═════╝ ╚══════╝ ╚═════╝             ║
║                                                               ║
║       Speaker Identification w. Log. Regression & GD          ║
║                     Final Project                             ║
║                                                               ║
║                   Created by: Evan Kim                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    print("Speakers:", get_speakers())

    while True:

        menu = input("[M]odify speakers [T]rain Model [R]un Model [S]tats [Q]uit: ").lower()

        if menu == "m":
            while True:
                choice = input("[A]dd a speaker, [C]lear speakers [D]one: ").lower()
                if choice == "a":
                    name = input("What is the name of this speaker: ")
                    base_dir = f"data/speakers/{name}"
                    os.makedirs(base_dir + "/raw", exist_ok=True)
                    os.makedirs(base_dir + "/processed", exist_ok=True)

                    print("\n" + "="*70)
                    print("PLEASE READ THE FOLLOWING TEXT ALOUD:")
                    print("="*70)
                    print(SPEECH)
                    print("="*70 + "\n")
                    record_audio(f"{base_dir}/raw/speech.wav")

                elif choice == "c":
                    print("Removing all speakers...")
                    reset_speakers("data/")

                elif choice == "d":
                    break

            print("Cleaning up audio...")
            reset_processed("data/speakers")
            print()
            process_dir("data/speakers")

        elif menu == "t":
            df = pd.read_csv("data/data.csv")
            print("Training...")
            weights_dict, _ = train(df, "speaker", ITER, ALPHA)

            print("Saving weights...")
            for speaker in weights_dict:
                np.save(f"weights/{speaker}.npy", weights_dict[speaker])

        elif menu == "r":
            speakers = get_speakers()
            if not speakers:
                print("Error: No speakers found. Please add speakers first.")
                continue

            print("Loading weights...")
            weights_dict = {}
            for speaker in speakers:
                try:
                    weights_dict[speaker] = np.load(f"weights/{speaker}.npy")
                    print(f"Loaded weights for speaker: {speaker}")
                except FileNotFoundError:
                    print(f"Warning: Weight file {speaker}.npy not found")

            if not weights_dict:
                print("Error: No weight files found. Please train the model first.")
                continue

            detect_realtime(model, weights_dict)

        elif menu == "s":
            display_accuracy_stats()

        elif menu == "q":
            break


if __name__ == "__main__":
    main()
