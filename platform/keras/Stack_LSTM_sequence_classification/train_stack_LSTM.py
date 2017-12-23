from models import *
from data_gen import *

data_dim = 16
timesteps = 8
num_classes = 2

train_gen = data_gen(timesteps,data_dim,100,train=True)
validation_gen = data_gen(timesteps,data_dim,100,train=False)

model = stack_LSTM(timesteps, data_dim,num_classes)

model.fit_generator(train_gen, steps_per_epoch=70, epochs=10, verbose=1, callbacks= callbacks, validation_data=validation_gen,validation_steps=10)
