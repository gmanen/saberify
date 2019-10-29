import os
import librosa
from io import open


def get_songs_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fma_small/000/')


def estimate_bpm(file):
    y, sr = librosa.load(file)
    onset_env = librosa.onset.onset_strength(y, sr=sr)

    return librosa.beat.tempo(onset_envelope=onset_env, sr=sr)


def generate_bpm_file():
    songs = [name for name in os.listdir(get_songs_dir())]
    songs_length = len(songs)

    for i in range(songs_length):
        song = songs[i]

        f = open("bpm.txt", "a")
        estimated_bpm = estimate_bpm(get_songs_dir() + song)

        f.write(str(estimated_bpm[0]) + "\n")
        f.close()


generate_bpm_file()
