import math
import os
from io import open
import json
import librosa
import struct
import numpy
import glob
import array


def get_data_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_songs_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bsaber/')


def get_data_file(num):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_' + str(num) + '.dat')


def get_data_files():
    return glob.glob(os.path.join(get_data_dir() + "*.dat"))


def load_info(dir):
    with open(dir + '/info.dat', 'r') as f:
        return json.load(f)


def extract_features(file, bands=128):
    sound_clip, s = librosa.load(file)

    melspec = librosa.feature.melspectrogram(sound_clip, n_mels=bands)

    return librosa.amplitude_to_db(melspec)


def dump_data(batches, bands=128):
    songs = [f.path for f in os.scandir(get_songs_dir()) if f.is_dir()]
    dataset_length = math.floor(len(songs) / batches)

    for i in range(batches):
        with open(get_data_file(i), 'wb') as dataFile:
            dataFile.write(struct.pack('i', bands))

            for j in range(dataset_length):
                song = songs[i * dataset_length + j]
                info = load_info(song)
                x = extract_features(song + '/' + info['_songFilename'], bands)
                y = info['_beatsPerMinute']

                dataFile.write(struct.pack('f', y))
                dataFile.write(struct.pack('i', x.shape[1]))

                for k in range(bands):
                    dataFile.write(struct.pack('f'*x.shape[1], *x[k].tolist()))


def load_data(file):
    with open(file, 'rb') as dataFile:
        bands = struct.unpack('i', dataFile.read(struct.calcsize('i')))[0]

        while True:
            read = dataFile.read(struct.calcsize('f'))

            if not read:
                break

            y = struct.unpack('f', read)[0]
            length = struct.unpack('i', dataFile.read(struct.calcsize('i')))[0]
            spec = []

            for i in range(bands):
                line = array.array('f')
                line.fromstring(dataFile.read(struct.calcsize('f'*length)))

                spec.append(line)

            x = numpy.array(spec)

            yield [x, y]


if __name__ == '__main__':
    dump_data(5)

    for data in load_data('./data_1.dat'):
        continue

    print('done')
