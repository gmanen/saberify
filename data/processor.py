import math
import os
from io import open
import json
import librosa
import numpy as np


def get_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bsaber/')


def load_info(dir):
    with open(dir + '/info.dat', 'r') as f:
        return json.load(f)


def windows(data, window_size):
    start = 0

    while start < len(data):
        yield math.floor(start), math.floor(start + window_size)
        start += (window_size / 2)


def extract_features(file, bands=128):
    sound_clip, s = librosa.load(file)

    melspec = librosa.feature.melspectrogram(sound_clip, n_mels=bands)

    return librosa.amplitude_to_db(melspec)


if __name__ == '__main__':
    songs = [f.path for f in os.scandir(get_data_dir()) if f.is_dir()]
    dataset_length = math.floor(len(songs) / 3)
    training_data = []

    for i in range(dataset_length):
        info = load_info(songs[i])
        label = info['_beatsPerMinute']

        y, sr = librosa.load(songs[i] + '/' + info['_songFilename'])
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        print(label)
        print(beat_frames.shape)
        spec = extract_features(songs[i] + '/' + info['_songFilename'])
        print(spec.shape)

    data_file = open('./data', 'wb+')
