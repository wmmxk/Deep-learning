'''
This code is to test whether you need to covert all the training labels to
categorical values once. Your concern is that the encoding might be different
'''
from keras.utils import np_utils

y_train = [0,1,5]
y_train = np_utils.to_categorical(y_train,10)
print(y_train)


y_train = [5,0,1]
y_train = np_utils.to_categorical(y_train,10)
print(y_train)

