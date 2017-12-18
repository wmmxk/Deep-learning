from keras.models import Model                                                                                
from keras.layers import Input, Conv2D,BatchNormalization, Activation,Dropout,Flatten,Dense                   
from keras.optimizers import RMSprop                                                                          
from keras.losses import categorical_crossentropy                                                             
from keras.initializers import glorot_uniform                                                                 
import keras.backend as K                                                                                     
                                                                                                              
def get_CNN(input_shape=(128,128,3), num_classes = 10):                                                       
    X_input = Input(shape = input_shape)                                                                      
    X = Conv2D(32,(3,3), padding = 'same',kernel_initializer = glorot_uniform(seed=0))(X_input)               
    X = Activation('relu')(X)                                                                                 
    X = Dropout(rate=0.5)(X)                                                                                            
                                                                                                              
    X = Conv2D(64,(3,3), padding = 'same',kernel_initializer = glorot_uniform(seed=0))(X)                
    X = Activation('relu')(X)                                                                                 
    X = Dropout(rate = 0.5)(X)                                                                                            
                                                                                                              
    X = Flatten()(X)                                                                                          
                                                                                                              
    X = Dense(1024,activation = 'relu',kernel_initializer = glorot_uniform(seed=0))(X)                        
    X = Dense(num_classes, activation = 'softmax', kernel_initializer = glorot_uniform(seed=0))(X)            
                                                                                                              
    model = Model(inputs = X_input, outputs = X )                                                             
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])                    
                                                                                                              
    return model 
