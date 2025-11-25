import os
import shutil


def reset_processed(base_dir):
    directories = os.listdir(path=base_dir)

    for directory in directories:
        print("Resetting directory", directory)
        person_dir = os.path.join(base_dir, directory)

        if not os.path.isdir(person_dir):
            continue

        processed_dir = os.path.join(person_dir, 'processed')
        for entry in os.listdir(processed_dir):
            os.remove(os.path.join(processed_dir, entry))


def reset_speakers(base_dir):
    shutil.rmtree(os.path.join(base_dir, "speakers"))


if __name__ == "__main__":
    reset_processed("data/speakers")
