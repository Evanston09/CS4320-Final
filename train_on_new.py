import wave
from reset_files import reset
import numpy as np
import os
import sys
import pandas as pd
import threading

from audio_processing import process_dir
import io
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
import matplotlib
import matplotlib.pylab as plt
import pyaudio
from utils import train, evaluate


ITER = 100000
ALPHA = 0.01

SPEECH = "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is , according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways. Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain. The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain. Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed. The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows. If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue."


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


def detect_realtime(model):
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
            grace = 0
            print("Someone talking")
            length += .032
            data.append(audio_chunk)
        else:
            if (grace < 32):
                print("Continuing grace applies")
                grace += 1
                length += .032

            else:
                print("Stopped Talking")
                data.clear()
                length = 0
        
        if length >= 5:
            print("5 Seconds reached")
            print("Saving speech")
            with wave.open("temp.wav", "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)

                for chunk in data:
                    wf.writeframes(chunk)

            print("Evaluating...")
            evaluate("temp.wav", get_speakers())

    print("Stopped the recording")

def main():
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
    )

    print("Welcome to the trainer for CS4320 Final Project")
    print("By: Evan Kim")

    print("Speakers:", get_speakers())

    while True:

        menu = input("[M]odify speakers [T]rain Model [R]un Model [Q]uit: ").lower()

        if menu == "m":
            choice = ""
            while choice != "d":
                choice = input("[A]dd a speaker, [D]one: ").lower()
                if choice == "a":
                    name = input("What is the name of this speaker: ")
                    dir = f"data/speakers/{name}/raw"
                    os.makedirs(dir, exist_ok=True)
                    print("Say this:\n" + SPEECH)
                    record_audio(f"{dir}/speech.wav")

            print("Cleaning up audio")
            reset("data/speakers")
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
            detect_realtime(model)

        elif menu == "q":
            break


if __name__ == "__main__":
    main()
