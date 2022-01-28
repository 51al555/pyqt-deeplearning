from tensorflow.keras.layers import Concatenate
import numpy as np
x = np.array([0, 1, 2, 4])
y = np.array([1, 1, 2, 2])
input_layer = Concatenate()([x, y])
print(input_layer)