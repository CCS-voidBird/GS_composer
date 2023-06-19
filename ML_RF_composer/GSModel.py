try:
    import keras
    from keras.models import Sequential
    from keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D
    from keras.layers import Dropout
    import tensorflow as tf
    import keras.metrics
except:
    print("This is not a GPU env.")
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import configparser

def testCNN(first_channels,second_channels,input_shape,config_path = "./configs/testCNN.ini",optimizer="rmsprop",lr=0.00001):
    lr = float(lr)
    model = Sequential()
    """
    Convolutional Layers
    """
    testconfig = configparser.ConfigParser()
    testconfig.read(config_path)
    model.add(Conv1D(first_channels, kernel_size=5, strides=3, padding='valid', activation='elu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(second_channels, kernel_size=3, strides=3, padding='valid', activation='elu'))
    model.add(MaxPooling1D(pool_size=2))

    # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
    model.add(Dropout(rate=0.2))

    model.add(Flatten())

    # Full connected layers, classic multilayer perceptron (MLP)
    for layers in range(4):
        model.add(Dense(8, activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
    try:
        adm = keras.optimizers.Adam(learning_rate=lr)
        rms = keras.optimizers.RMSprop(learning_rate=lr)
        sgd = keras.optimizers.SGD(learning_rate=lr)
    except:
        adm = keras.optimizers.Adam(lr=lr)
        rms = keras.optimizers.RMSprop(lr=lr)
        sgd = keras.optimizers.SGD(lr=lr)

    optimizers = {"rmsprop": rms,
                  "Adam": adm,
                  "SGD": sgd}

    model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error")

    """
    Optimizers: Adam, RMSProp, SGD 
    """

    return model

def CNN(n_layers,n_units,input_shape,optimizer="rmsprop",lr=0.00001):
    lr = float(lr)
    model = Sequential()
    """
    Convolutional Layers
    """
    model.add(Conv1D(64, kernel_size=5, strides=3, padding='valid', activation='elu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, kernel_size=3, strides=3, padding='valid', activation='elu'))
    model.add(MaxPooling1D(pool_size=2))

    # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
    model.add(Dropout(rate = 0.2))

    model.add(Flatten())

    # Full connected layers, classic multilayer perceptron (MLP)
    for layers in range(n_layers):
        model.add(Dense(n_units,activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear")) # The output layer uses a linear function to predict traits.
    try:
        adm = keras.optimizers.Adam(learning_rate=lr)
        rms = keras.optimizers.RMSprop(learning_rate=lr)
        sgd = keras.optimizers.SGD(learning_rate=lr)
    except:
        adm = keras.optimizers.Adam(lr=lr)
        rms = keras.optimizers.RMSprop(lr=lr)
        sgd = keras.optimizers.SGD(lr=lr)

    optimizers = {"rmsprop":rms,
                 "Adam": adm,
                 "SGD": sgd}

    model.compile(optimizer=optimizers[optimizer],loss="mean_squared_error")

    """
    Optimizers: Adam, RMSProp, SGD 
    """

    return model

def DeepGS(input_shape,n_layers=0,n_units=32,optimizer="SGD",lr=0.01):
    print("If need to specify MLP parameters for DeepGS model, modify MODEL file instead.")
    lr = float(lr)
    model = Sequential()
    model.add(Conv1D(8, kernel_size=18, strides=1, padding='valid', activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=4,strides=4))
    model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dropout(rate=0.1))
    model.add(Dense(32,activation="relu"))
    model.add(Dropout(rate=0.05))
    model.add(Dense(1,activation="linear"))

    try:
        adm = keras.optimizers.Adam(learning_rate=lr)
        rms = keras.optimizers.RMSprop(learning_rate=lr)
        sgd = keras.optimizers.SGD(learning_rate=lr)
    except:
        adm = keras.optimizers.Adam(lr=lr)
        rms = keras.optimizers.RMSprop(lr=lr)
        sgd = keras.optimizers.SGD(lr=lr)

    optimizers = {"rmsprop":rms,
                 "Adam": adm,
                 "SGD": sgd}

    model.compile(optimizer=optimizers[optimizer],loss="mean_squared_error")

    return model


def TDCNN(n_layers,n_units,input_shape,optimizer="rmsprop",lr=0.00001):
    n_factors = input_shape[1]
    lr = float(lr)
    model = Sequential()
    """
    Convolutional Layers
    """
    model.add(Conv2D(64,kernel_size=(5,n_factors),strides=(3,1),padding='valid',activation='elu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,1)))

    model.add(Conv2D(128, kernel_size=(3,1), strides=(3,1), padding='valid',activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,1)))

    # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
    model.add(Dropout(rate = 0.2))

    model.add(Flatten())

    # Full connected layers, classic multilayer perceptron (MLP)
    for layers in range(n_layers):
        model.add(Dense(n_units,activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear")) # The output layer uses a linear function to predict traits.
    try:
        adm = keras.optimizers.Adam(learning_rate=lr)
        rms = keras.optimizers.RMSprop(learning_rate=lr)
        sgd = keras.optimizers.SGD(learning_rate=lr)
    except:
        adm = keras.optimizers.Adam(lr=lr)
        rms = keras.optimizers.RMSprop(lr=lr)
        sgd = keras.optimizers.SGD(lr=lr)

    optimizers = {"rmsprop":rms,
                 "Adam": adm,
                 "SGD": sgd}

    model.compile(optimizer=optimizers[optimizer],loss="mean_squared_error")

    """
    Optimizers: Adam, RMSProp, SGD 
    """

    return model

def MLP(n_layers,n_units,input_shape,optimizer="rmsprop",lr=0.00001):
    model = Sequential()
    model.add(Dense(n_units, activation="elu",input_shape=input_shape))
    for layers in range(n_layers-1):
        model.add(Dense(n_units, activation="elu"))
    #model.add(Dropout(0.2))

    model.add(Dense(1, activation="linear"))

    try:
        adm = keras.optimizers.Adam(learning_rate=lr)
        rms = keras.optimizers.RMSprop(learning_rate=lr)
        sgd = keras.optimizers.SGD(learning_rate=lr)
    except:
        adm = keras.optimizers.Adam(lr=lr)
        rms = keras.optimizers.RMSprop(lr=lr)
        sgd = keras.optimizers.SGD(lr=lr)

    optimizers = {"rmsprop":rms,
                 "Adam": adm,
                 "SGD": sgd}

    model.compile(optimizer=optimizers[optimizer],loss="mean_squared_error")

    return model

def RF(config = None,specific=True,n_features = 500,n_estimators = 200):
    if specific == True:
        model = RandomForestRegressor(n_jobs=-1, random_state=0, criterion="mse", oob_score=False, verbose=1,max_features=n_features,
                                      n_estimators=n_estimators)
        return model
    else:
        rm_config = {x: int(config["RM"][x]) for x in config["RM"].keys()}
        try:
            model = RandomForestRegressor(n_jobs=-1, random_state=0, criterion="mse", oob_score=False, verbose=1,
                                          **rm_config)
        except:
            print("Cannot find the config file 'MLP_parameters.ini")
            model = RandomForestRegressor(n_jobs=-1, random_state=0, criterion="mse", oob_score=False, verbose=1,
                                          n_estimators=2000)

        return model

METHODS = {
    "RF": RF,
    "MLP": MLP,
    "CNN": CNN,
    "TDCNN": TDCNN,
    "DeepGS": DeepGS
}

def main():
    model = CNN(n_layers=3,n_units=8,input_shape=[26084,4])
    tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()
