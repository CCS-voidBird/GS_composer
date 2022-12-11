try:
    import keras
    from keras.models import Sequential
    from tensorflow.keras import layers
    from keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D
    from keras.layers import Dropout
    from tensorflow.keras.utils import to_categorical
    import tensorflow as tf
    import keras.metrics
except:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D, Dropout
        import tensorflow as tf

        print("Use tensorflow backend keras module")
    except:
        print("This is not a GPU env.")
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import numpy as np
import configparser
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


####################
"""
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

####################
"""

class NN:

    def __init__(self,args):
        self.name = "NN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def model(self, input_shape, args, optimizer="rmsprop", lr=0.00001):
        pass

class Transformer(NN):
    
    #############################
    #Need work!!!!!!!!!!#
    ###########################

    def __init__(self,args):
        super(Transformer,self).__init__(args)
        self.name = "Transformer"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        geno.replace(0.01, 0, inplace=True)
        #geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        output_dim=4
        lr = float(lr)
        model = Sequential()
        # Add an Embedding layer expecting input vocab of size sequence length, and
        # output embedding dimension of size 64.

        #model.add(layers.Input(shape=input_shape, dtype="float32"))
        #model.add(layers.Embedding(input_dim=3, output_dim=output_dim))
        # Add a LSTM layer with 128 internal units.
        model.add(layers.Bidirectional(layers.LSTM(64)))

        # Add a Dense layer with defined units.
        model.add(layers.Dense(args.width))

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


class RNN(NN):

    #############################
    # Need work!!!!!!!!!!#
    ###########################
    # super init function with RNN
    def __init__(self,args):
        NN.__init__(self,args)
        self.name = "RNN"

    def model_name(self):
        # get class name
        return self.__class__.__name__

    def data_transform(self, geno, pheno, anno=None, pheno_standard=False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        geno.replace(0.01, 0, inplace=True)
        # geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno, pheno

    def model(self, input_shape, args, optimizer="rmsprop", lr=0.00001):
        embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        output_dim = 4
        lr = float(lr)
        model = Sequential()
        # Add an Embedding layer expecting input vocab of size sequence length, and
        # output embedding dimension of size 64.
        model.add(layers.Input(shape=input_shape, dtype="float32"))

        # model.add(layers.Embedding(input_dim=3, output_dim=output_dim))

        # Add a LSTM layer with 128 internal units.
        model.add(layers.Bidirectional(layers.LSTM(64)))

        # Add a Dense layer with defined units.
        model.add(layers.Dense(args.width))

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


class DCNN():
    """
    double CNN, esitmate additive SNP alleles and heterozygous SNP alleles
    """
    def __init__(self):
        self.name = "Deep Residual CNN"

    def model_name(self):
        #get class name
        return self.__class__.__name__


    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Deep Residual CNN MODEL as training method")
        geno = decoding(geno)
        geno1 = geno
        geno2 = geno.mask(geno != 1,0)
        #overlap geno1 and geno2 to one matrix
        #geno1 = np.expand_dims(geno1, axis=2)
        #geno2 = np.expand_dims(geno2, axis=2)
        geno = np.stack((geno1,geno2),axis=2)

        print("The transformed SNP shape:", geno.shape)
        print(geno[0:10,0:10,1])
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        lr = float(lr)
        #model = Sequential()
        """
        Convolutional Layers
        """
        #model.add(layers.Input(shape=input_shape, dtype="float32"))
        #model.add(layers.BatchNormalization())
        input = layers.Input(shape=input_shape)
        input1 = layers.Conv1D(32, 7, strides=3, padding="same")(input)
        input1 = layers.Activation("elu")(input1)
        input1 = layers.MaxPooling1D(3)(input1)
        #input1 = layers.BatchNormalization()(input)
        input1 = layers.Conv1D(32,1,padding="same")(input1)
        input1 = layers.Activation("relu")(input1)

        X = layers.Conv1D(32, 3,padding="same")(input1)
        X = layers.LeakyReLU(alpha=0.3)(X)
        X = layers.BatchNormalization()(X)
        #first_layer = layers.Activation("linear", trainable=False)(X)
        X = layers.Conv1D(32, 3, padding="same")(X)
        X = layers.LeakyReLU(alpha=0.3)(X)
        X = layers.BatchNormalization()(X)
        #X = layers.BatchNormalization()(X)

        residual1 = layers.Add()([X, input1])
        #X = layers.LeakyReLU(alpha=0.3)(residual1)

        input2 = layers.MaxPooling1D(3)(residual1)
        input2 = layers.Conv1D(64, 1, activation="relu")(input2)
        X = layers.Conv1D(64, 3, padding="same")(input2)
        X = layers.LeakyReLU(alpha=0.3)(X)
        #X = layers.BatchNormalization()(X)

        #second_layer = layers.Activation("linear", trainable=False)(X)

        X = layers.Conv1D(64, 3, padding="same")(X)
        X = layers.LeakyReLU(alpha=0.3)(X)
        #X = layers.BatchNormalization()(X)

        residual2 = layers.Add()([X, input2])
        #X = layers.LeakyReLU(alpha=0.3)(residual)
        #X = Residual(32, 3)(X)
        #X = Residual(32, 3)(X)
        #X = Residual(32, 3)(X)
        #X = layers.BatchNormalization()(X)
        X = layers.MaxPooling1D(2)(residual2)
        X = layers.Flatten()(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(args.width)(X)
        X = layers.Activation("relu")(X)
        #X = layers.Dropout(0.2)(X)
        output = layers.Dense(1, activation="linear")(X)
        #model.add(Conv1D(32, kernel_size=5, strides=3, padding='same',input_shape=input_shape))
        #model.add(layers.Activation('relu'))
        #model.add(Residual(32, 3))
        #model.add(Residual(32, 3))
        #model.add(Residual(32, 3))
        #model.add(layers.BatchNormalization())
        #model.add(Conv1D(128, kernel_size=3, strides=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        #model.add(Flatten())
        #model.add(Dropout(rate=0.2))
        #model.add(layers.BatchNormalization())
        # Full connected layers, classic multilayer perceptron (MLP)
        #for i in range(args.depth): model.add(Dense(args.width, activation="relu"))
        #model.add(Dense(512))
        #model.add(Dropout(0.2))
        #model.add(layers.Activation("linear"))  # The output layer uses a linear function to predict traits.
        model = keras.Model(inputs=input, outputs=output,name="DCNN")
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


class DoubleCNN(NN):
    """
    double CNN, esitmate additive SNP alleles and heterozygous SNP alleles
    """
    def __init__(self,args):
        super(DoubleCNN, self).__init__(args)
        self.name = "Double channel residual CNN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__


    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Duo (Double) CNN MODEL as training method")
        #geno = decoding(geno)
        #geno = np.expand_dims(geno, axis=2)
        geno1 = geno
        geno1 = decoding(geno1)
        geno1 = np.expand_dims(geno1, axis=2)
        geno2 = geno.mask(geno != 1,0)
        geno2 = decoding(geno2)
        geno2 = np.expand_dims(geno2, axis=2)
        #geno = np.stack((geno1,geno2),axis=2)
        #geno = np.expand_dims(geno, axis=3)
        print("The transformed SNP shape:", geno1.shape,geno2.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return [geno1,geno2],pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.0001):

        input1 = layers.Input(shape=input_shape)
        X1 = layers.Conv1D(64, kernel_size=25, strides=3, padding='same')(input1)
        X1 = layers.Activation('relu')(X1)
        X1 = layers.MaxPooling1D(pool_size=2)(X1)
        X1 = layers.Conv1D(128, kernel_size=3, strides=3, padding='same')(X1)
        X1 = layers.Activation('relu')(X1)
        X1 = layers.MaxPooling1D(pool_size=2)(X1)

        input2 = layers.Input(shape=input_shape)
        X2 = layers.Conv1D(64, kernel_size=25, strides=3, padding='same')(input2)
        X2 = layers.Activation('relu')(X2)
        X2 = layers.MaxPooling1D(pool_size=2)(X2)
        X2 = layers.Conv1D(128, kernel_size=3, strides=3, padding='same')(X2)
        X2 = layers.Activation('relu')(X2)
        X2 = layers.MaxPooling1D(pool_size=2)(X2)

        X = layers.concatenate([X1, X2], axis=-1)
        X = layers.Flatten()(X)
        X = layers.Dropout(rate=0.2)(X)

        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width, downsample=(i % 2 != 0 & self.args.residual))

        output = layers.Dense(1, activation="linear")(X)
        model = keras.Model(inputs=[input1,input2], outputs=output)

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
        input = layers.Input(shape=input_shape)
        X = layers.Conv1D(64, kernel_size=5, strides=3, padding='same', activation='elu')(input)
        X = layers.MaxPooling1D(pool_size=2)(X)
        X = layers.Conv1D(128, kernel_size=3, strides=3, padding='same', activation='elu')(X)
        X = layers.MaxPooling1D(pool_size=2)(X)

        X = layers.Dropout(rate=0.2)(X)
        X = layers.Flatten()(X)

        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width,activation=layers.ELU(),downsample=(i%2 != 0 & self.args.residual))
        X = layers.Dropout(rate=0.2)(X)
        output = layers.Dense(1, activation="linear")(X)
        model = keras.Model(inputs=input, outputs=output)

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

    
class BCNN():

    def __init__(self):
        self.name = "Binary CNN"

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self, geno, pheno, anno=None,pheno_standard = False):
        print("USE Binary CNN MODEL as training method")
        geno = decoding(geno)
        geno.replace(0.01, 3, inplace=True)
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


class MLP():

    def __init__(self):
        self.name = "MLP"

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

class LNN(NN):

    def __init__(self,args):
        super(LNN,self).__init__(args)
        self.name = "Local NN"
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

        input = layers.Input(shape=input_shape)
        X = layers.ZeroPadding1D(padding=(0, input_shape[1] // 5))(input)
        X = layers.LocallyConnected1D(128, kernel_size=5, strides=5, padding='valid', activation='elu')(X)
        X = layers.Conv1D(256, kernel_size=20, strides=2, padding='valid', activation='elu')(X)

        #X = layers.LocallyConnected1D(128, kernel_size=3, strides=3, padding='valid', activation='elu')(X)

        #X = layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
        X = layers.Flatten()(X)
        X = layers.Dropout(0.2)(X)
        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width,activation=layers.ELU(),downsample=(i%2 != 0 & self.args.residual))

        output = layers.Dense(1, activation="linear")(X)
        model = keras.Model(inputs=input, outputs=output)

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


def RF(config = None,specific=False,n_features = 500,n_estimators = 200):
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

class AttentionCNN(NN):

    def __init__(self,args):
        super(AttentionCNN,self).__init__(args)
        self.name = "Attention CNN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):

        print("USE Attention CNN MODEL as training method")
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        #pos = np.arrays(range(geno.shape[1]))
        #pos = np.expand_dims(pos, axis=0)
        print("The transformed SNP shape:", geno.shape)

        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        # init Q,K,V
        depth = args.depth
        input1 = layers.Input(shape=input_shape,name="input_layer_1")

        X = layers.ZeroPadding1D(padding=(0, input_shape[1]//10))(input1)
        V = layers.LocallyConnected1D(1,10,strides=10, activation="relu",padding="valid",use_bias=False)(X)
        #Q = PositionalEncoding(position=input_shape[0], d_model=input_shape[1])(V)
        V = layers.Reshape((input_shape[0],))(V)

        Pos = PositionalEncoding(position=input_shape[0], d_model=input_shape[-1])(V)

        # Q,K,V 1D Conv
        #Q_encoding = layers.Conv1D(filters=16, kernel_size=1, strides=1, padding="same", activation="relu")(Pos)

        # Attention
        pos_attention = layers.Attention(use_scale=True)([Pos,Pos])
        #Q_attention = layers.GlobalAvgPool1D()(Q_encoding)
        #QV_attention = layers.GlobalAvgPool1D()(QV_attention)
        # Concat
        M = layers.Concatenate()([V,pos_attention])
        # Residual Dense

        M = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding="same", activation="elu")(M)
        M = layers.GlobalAvgPool1D()(M)

        while depth > 0:
            M = residual_fl_block(input=M, width=self.args.width, downsample=(depth % 2 == 0 & self.args.residual))
            depth -= 1
        QV_output = layers.Dense(1, activation="linear")(M)

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

        model = keras.Model(inputs=input1, outputs=QV_output)
        model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error")

        #QK = layers.Dot(axes=[2, 2])([Q_encoding, K_encoding])
        #QK = layers.Softmax(axis=-1)(QK)
        #QKV = layers.Dot(axes=[2, 1])([QK, V_encoding])
        #QKV = layers.Flatten()(QKV)
        #QKV = layers.Dense(1, activation="linear")(QKV)

        return model

MODELS = {
    "MLP": MLP,
    "Numeric CNN": NCNN,
    "Binary CNN": BCNN,
    "Test CNN":Transformer,
    "Duo CNN": DCNN,
    "Double CNN": DoubleCNN,
    "Attention CNN": AttentionCNN,
    "ResMLP": ResMLP,
    "LNN": LNN,
}

def main():
    print("Main function from ClassModel.py")
    #tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()
