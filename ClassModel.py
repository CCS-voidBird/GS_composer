try:
    import keras
    from keras.models import Sequential
    from tensorflow.keras import layers
    from keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D
    from keras.layers import Dropout
    from tensorflow.keras.utils import to_categorical
    import tensorflow as tf
    import keras.metrics
    from keras import layers
    
except:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D, Dropout
        import tensorflow as tf
        from keras import layers
        print("Use tensorflow backend keras module")
    except:
        print("This is not a GPU env.")
from scipy import stats
import numpy as np
from Functions import *
from tensorflow.keras.layers import Layer
from CustomLayers import *
tf.config.experimental_run_functions_eagerly(True)
# Define the residual block as a new layer


class Residual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel
        #self.Conv1D = layers.Conv1D(self.channels_in, self.kernel, padding="same")
        #self.Activation = layers.Activation("relu")
    def call(self, x):
        # the residual block using Keras functional API
        first_layer = layers.Activation("linear", trainable=False)(x)
        x = Conv1D(self.channels_in, self.kernel, padding="same")(first_layer)
        x = layers.Activation("relu")(x)
        x = Conv1D(self.channels_in, self.kernel, padding="same")(x)
        residual = layers.Add()([x, first_layer])
        x = layers.Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class NN:

    def __init__(self,args):
        self.name = "NN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def model(self, input_shape, args, optimizer="rmsprop", lr=0.00001):
        pass


class NCNN(NN):

    def __init__(self,args):
        super(NCNN,self).__init__(args)
        self.name = "Numeric CNN"
        self.args = args


    def model_name(self):
        #get class name
        if self.args.residual is True:
            return "Res"+self.__class__.__name__
        else:
            return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        lr = float(lr)

        """
        Convolutional Layers
        model = Sequential()
        model.add(Conv1D(64, kernel_size=5, strides=3, padding='valid', activation='elu',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(128, kernel_size=3, strides=3, padding='valid', activation='elu'))
        model.add(MaxPooling1D(pool_size=2))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        model.add(Dropout(rate=0.2))

        model.add(Flatten())

        # Full connected layers, classic multilayer perceptron (MLP)
        for i in range(args.depth):
            model.add(Dense(args.width, activation="elu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
        """
        input1 = layers.Input(shape=input_shape)
        X = layers.Conv1D(64, kernel_size=5, strides=3, padding='same', activation='elu')(input1)
        X = layers.MaxPooling1D(pool_size=2)(X)
        X = layers.Conv1D(128, kernel_size=3, strides=3, padding='same', activation='elu')(X)
        X = layers.MaxPooling1D(pool_size=2)(X)

        X = layers.Dropout(rate=0.2)(X)

        X = layers.Flatten()(X)

        # Full connected layers, classic multilayer perceptron (MLP)
        for layer_index in range(args.depth):
            X = Dense(args.width, activation="elu")(X)
        X = layers.Dropout(rate=0.2)(X)
        output = layers.Dense(1, activation="linear")(X)

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
        model = keras.Model(inputs=input1, outputs=output)
        model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error")

        """
        Optimizers: Adam, RMSProp, SGD 
        """

        return model

    
class BCNN(NN):

    def __init__(self,args):
        super(BCNN,self).__init__(args)
        self.name = "Binary CNN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self, geno, pheno, anno=None,pheno_standard = False):
        print("USE Binary CNN MODEL as training method")
        geno = decoding(geno)
        #geno.replace(0.01, 3, inplace=True)
        geno = to_categorical(geno)
        print("The transformed SNP shape:",geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
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
        model.add(Dropout(rate=0.2))

        model.add(Flatten())

        # Full connected layers, classic multilayer perceptron (MLP)
        for layers in range(args.depth):
            model.add(Dense(args.width, activation="elu"))
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


class MLP(NN):

    def __init__(self,args):
        super(MLP,self).__init__(args)
        self.name = "MLP"
        self.args = args


    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        #geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        model = Sequential()
        model.add(Dense(args.width, activation="elu", input_shape=input_shape))
        for layers in range(args.depth - 1):
            model.add(Dense(args.width, activation="elu"))
        # model.add(Dropout(0.2))

        model.add(Dense(1, activation="linear"))

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

        return model

class ResMLP(NN):

    def __init__(self,args):
        super(ResMLP,self).__init__(args)
        self.name = "Residual MLP"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        #geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):

        # model.add(Dropout(0.2))

        input = layers.Input(shape=input_shape)
        X = layers.BatchNormalization()(input)

        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width,activation=layers.ELU(),downsample=(i % 2 != 0 & self.args.residual))

        output = layers.Dense(1, activation="linear")(X)

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

        model = keras.Model(inputs=input, outputs=output)

        model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error")

        return model

MODELS = {
    "MLP": MLP,
    "Numeric CNN": NCNN,
    "Binary CNN": BCNN,
}

def main():
    print("Main function from ClassModel.py")
    #tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()
