import os
import time
import numpy as np
import tensorflow as tf
from HypercomplexKeras.Hyperdense import HyperDense
from keras import Sequential
from keras.src.layers import Dense, Activation

# This code is just for testing purposes

# check if GPU is available:
mode = "GPU"

if mode == "GPU":
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
elif mode == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))

start_time = time.time()

# data:
x_train = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.dtype(float))
y_train = np.array([[0], [1], [1], [0]])

# create model:
model = Sequential()
model.add(HyperDense(2000))
model.add(HyperDense(10000))
model.add(HyperDense(2000))
model.add(HyperDense(3000))
model.add(HyperDense(4000))
# model.add(Dense(num_neurons))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.predict(x_train, verbose=0)

model.summary()

sgd = tf.keras.optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=1000, verbose=1)

y_predict = model.predict(x_train)
y_predict_quantized = np.round(y_predict).astype(int)

end_time = time.time()

print(f"Time for {mode}: {(end_time - start_time):.2f} seconds")