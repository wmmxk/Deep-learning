from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def stack_LSTM(time_steps,data_dim, num_classes):

    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(time_steps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32

    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


                                                                                                              
callbacks = [EarlyStopping(monitor='val_acc',                                                                 
                           patience=8,                                                                        
                           verbose=1,                                                                         
                           min_delta=1e-4,                                                                    
                           mode='max'),                                                                       
             ReduceLROnPlateau(monitor='val_acc',                                                             
                               factor=0.1,                                                                    
                               patience=4,                                                                    
                               verbose=1,                                                                     
                               epsilon=1e-4,                                                                  
                               mode='max'),                                                                   
             ModelCheckpoint(monitor='val_acc',                                                               
                             filepath='./best_weights.hdf5',                             
                             save_best_only=True,                                                             
                             save_weights_only=True,                                                          
                             mode='max')] 


if __name__=="__main__":
    data_dim = 16
    time_steps = 8
    num_classes = 2
    stack_LSTM(time_steps, data_dim, num_classes)
