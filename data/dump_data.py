import math
import os
from io import open
import librosa
import struct
import numpy
import glob


def get_data_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_songs_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fma_small/200/')


def get_song_path(song):
    return get_songs_dir() + song


def get_data_file(num):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_' + str(num) + '.dat')


def get_data_files():
    return glob.glob(os.path.join(get_data_dir() + "*.dat"))


def extract_features(file, bands=128):
    sound_clip, s = librosa.load(file)

    melspec = librosa.feature.melspectrogram(sound_clip, n_mels=bands)

    return librosa.amplitude_to_db(melspec)


def estimate_bpm(file):
    y, sr = librosa.load(file)
    onset_env = librosa.onset.onset_strength(y, sr=sr)

    return librosa.beat.tempo(onset_envelope=onset_env, sr=sr)


def dump_data(batches, bands=128):
    songs = [name for name in os.listdir(get_songs_dir())]
    songs_length = math.floor(len(songs) / batches)

    for i in range(batches):
        with open(get_data_file(i), 'wb') as data_file:
            data_file.write(struct.pack('i', bands))

            for j in range(songs_length):
                song = songs[i * songs_length + j]
                x = extract_features(get_song_path(song), bands)
                y = float(estimate_bpm(get_song_path(song))[0])

                if len(x[0]) == 1291:
                    fill = numpy.zeros((128, 2), dtype=int)
                    x = numpy.append(x, fill, axis=1)

                data_file.write(struct.pack('f', y))
                data_file.write(struct.pack('i', x.shape[1]))

                for k in range(bands):
                    data_file.write(struct.pack('f'*x.shape[1], *x[k].tolist()))


if __name__ == '__main__':
    dump_data(1)

    print('done')
