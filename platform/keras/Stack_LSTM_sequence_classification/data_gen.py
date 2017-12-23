import numpy as np


def data_gen(timesteps,data_dim,batch_size,train=True):
    x_train = np.random.random((7000, timesteps, data_dim))
    y_train = np.eye(2)[[int(x>6) for x in np.sum(np.sum(x_train,axis=1),axis = 1)/10]]

    # Generate dummy validation data
    x_val = np.random.random((3000, timesteps, data_dim))
    y_val = np.eye(2)[[int(x>6) for x in np.sum(np.sum(x_val,axis=1),axis = 1)/10]]
    
    X = x_train if train else x_val
    Y = y_train if train else y_val

    n = X.shape[0]
    while True:
        for start in range(0,n,n//batch_size):
            end = min(start + batch_size, n)
            yield (X[start:end], Y[start:end])


if __name__=="__main__":
    timesteps = 10
    data_dim = 20
    train_gen = data_gen(timesteps,data_dim,100)
    x, y = next(train_gen)
    print("x shape: ",x.shape)
