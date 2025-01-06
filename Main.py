import os
import time
import numpy as np
import tensorflow as tf
from HypercomplexKeras.Hyperdense import HyperDense
from keras import Sequential
from keras.src.layers import Dense, Activation
from keras.src.optimizers import Adam

# This code is just for testing purposes

# check if GPU is available:
mode = "GPU"

if mode == "GPU":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
elif mode == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))

# Initialize MirroredStrategy for multi-GPU setup
strategy = tf.distribute.MirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# data:
def generate_dummy_data(num_samples):
    x_train = np.random.rand(num_samples, 4)
    y_train = np.array([[1] if x[0] > 0.5 > x[1] else [0] for x in x_train])
    return x_train, y_train

x_train, y_train = generate_dummy_data(100000)

# Open a strategy scope to run model creation and compilation
with strategy.scope():
    model = Sequential()
    model.add(HyperDense(2000))
    model.add(HyperDense(10000))
    model.add(HyperDense(2000))
    model.add(HyperDense(2000))
    model.add(HyperDense(2000))
    model.add(HyperDense(3000))
    model.add(HyperDense(4000))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.build(input_shape=(None, 4))

print("Starting training...")
start_time = time.time()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=256, verbose=1)

# Predict with the model
y_predict = model.predict(x_train)
y_predict_quantized = np.round(y_predict).astype(int)

end_time = time.time()

print(f"Time for {mode}: {(end_time - start_time):.2f} seconds")
