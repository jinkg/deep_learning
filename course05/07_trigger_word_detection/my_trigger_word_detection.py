from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from pydub import AudioSegment

import numpy as np

from detection_utils import *

X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


def model(input_shape):
    """
        Function creating the model's graph in Keras.

        Argument:
        input_shape -- shape of the model's input data (using Keras conventions)

        Returns:
        model -- Keras model instance
    """
    X_input = Input(shape=input_shape)

    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)

    model = Model(inputs=X_input, outputs=X)

    return model


model = model((Tx, n_freq))

model.summary()

model = load_model('./models/tr_model.h5')

# opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
# model.fit(X, Y, batch_size = 5, epochs=1)

loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


def detect_triggerword(filename):
    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions


chime_file = "audio_examples/chime.wav"


def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0

    audio_clip.export("chime_output.wav", format='wav')


filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)

# filename = "./raw_data/dev/2.wav"
# prediction = detect_triggerword(filename)
# chime_on_activate(filename, prediction, 0.5)
#
#
# def preprocess_audio(filename):
#     # Trim or pad audio segment to 10000ms
#     padding = AudioSegment.silent(duration=10000)
#     segment = AudioSegment.from_wav(filename)[:10000]
#     segment = padding.overlay(segment)
#     # Set frame rate to 44100
#     segment = segment.set_frame_rate(44100)
#     # Export as wav
#     segment.export(filename, format='wav')
#
#
# your_filename = "audio_examples/my_audio.wav"
# preprocess_audio(your_filename)
# chime_threshold = 0.5
# prediction = detect_triggerword(your_filename)
# chime_on_activate(your_filename, prediction, chime_threshold)
