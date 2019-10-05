import soundfile as sf

filename = "samples/sample_1.wav"

data, fs = sf.read(filename, dtype='float32')

print(fs)
