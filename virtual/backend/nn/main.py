import numpy as np 
import nnfs 
from tensorflow.keras.datasets import fashion_mnist

from Layer import Dense
from ActivationFunction import ReLU, Linear
from Optimizer import Adam 
from Loss import MeanSquaredError
from Model import Model 
from Accuracy import Regression

nnfs.init() 

EPOCHS = 10 
BATCH_SIZE = 128

(x, y), (x_test, y_test) = fashion_mnist.load_data()
keys = np.array(range(x.shape[0]))
np.random.shuffle(keys) 
x = x[keys] 
y = y[keys] 

x = (x.reshape(x.shape[0], -1).astype(np.float32) - 127.5) / 127.5
x = (x.reshape(x.shape[0], -1).astype(np.float32) - 127.5) / 127.5

steps = x.shape[0] // BATCH_SIZE

if steps * BATCH_SIZE < BATCH_SIZE: 
    steps += 1 

# reshape to be a list of list 
# inner list contains one output (0 or 1) per each output neuron 
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = Model() 
model.add(Dense(1, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ReLU())
model.add(Dense(512, 512))
model.add(ReLU())
model.add(Dense(512, 1))
model.add(Linear())
model.set(
    loss=MeanSquaredError(), 
    optimizer=Adam(learning_rate=0.005, decay=1e-3), 
    accuracy=Regression()
)

model.compile()
model.train(x, y, epochs=EPOCHS, print_every=100, 
            validation_data=(x_test, y_test), batch_size=BATCH_SIZE)
model.evaluate(x_test, y_test)

parameters = model.get_parameters() 
