from _init_paths import *
from data_gen.data_generator import *
from data_gen.data_generator_correct import *
from models.CNN import *
import sys
args = sys.argv


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
if args[1]=="correct":
    tr_generator = data_generator_correct(os.path.join(project_dir,"data"), train =True,batch_size = 20)    
    te_generator = data_generator_correct(os.path.join(project_dir,"data"), train = False, batch_size = 8)  
else:
    tr_generator = data_generator(os.path.join(project_dir,"data"), train =True,batch_size = 20)    
    te_generator = data_generator(os.path.join(project_dir,"data"), train = False, batch_size = 8)  

imgs, labels = next(tr_generator)

input_shape = imgs.shape[1:]
num_classes = labels.shape[1]
model = get_CNN(input_shape = input_shape, num_classes = num_classes)

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
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max'),
             TensorBoard(log_dir='logs')]


# steps_per_epoch should be equal len(images_tr)//batch_size
model.fit_generator(generator = tr_generator, 
        steps_per_epoch = 10,
        verbose = 1,
        epochs = 8,
        callbacks = callbacks,
        validation_data = te_generator,
        validation_steps = 4)
