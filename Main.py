import os
import time

import numpy as np
import tensorflow as tf
from HypercomplexKeras.Hyperdense import HyperDense
from keras import Sequential
from keras.src.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.src.optimizers import Adam

# This code is just for testing purposes

# check if GPU is available:
mode = "GPU"

num_threads = os.cpu_count()

print(f"Number of threads: {num_threads}")

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
os.environ["OMP_NUM_THREADS"] = str(num_threads)

if mode == "GPU":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
elif mode == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))

# Initialize MirroredStrategy for multi-GPU setup
strategy = tf.distribute.MirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")


# data:
num_samples = 100000
img_height, img_width = 64, 64
x_train = np.random.rand(num_samples, img_height, img_width, 3)
y_train = np.random.randint(0, 2, num_samples)

# Open a strategy scope to run model creation and compilation
with strategy.scope():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.build()

print("Starting training...")
start_time = time.time()

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=512, verbose=1)

# Predict with the model
y_predict = model.predict(x_train)
y_predict_quantized = np.round(y_predict).astype(int)

end_time = time.time()

print(f"Time for {mode}: {(end_time - start_time):.2f} seconds")
