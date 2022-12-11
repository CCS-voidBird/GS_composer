
from Functions import *
from ClassModel import *
import pandas as pd
import matplotlib.pyplot as plt
try:
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
except:
    print("This a CPU-only platform.")
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import platform
from datetime import datetime
from sklearn.metrics import mean_squared_error
import configparser
import gc
import os
##############################################################
##########Training requirement################################
##Training by para-sets -- convolutional act function + full connected act function + optimizer + learningRate###
##Output format: an table with mean accuracy for each para set; A density plot for each accuracy##########
#################################################################
#Test command (Local): python GS_base.py --config ./test_config
CNNs = ["CNN","TDCNN","DeepGS"]
PATIENCE = 100
"""
GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"

TRAIN_PATH = "E:/learning resource/PhD/sugarcane/2015_TCHBlup_2000.csv"
VALID_PATH = "E:/learning resource/PhD/sugarcane/2016_TCHBlup_2000.csv"
"""
def get_args():
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument('--ped', type=str, help="PED-like file name", required=True)
    req_grp.add_argument('-pheno', '--pheno', type=str, help="Phenotype file.", required=True)
    req_grp.add_argument('-mpheno', '--mpheno', type=int, help="Phenotype columns (start with 1).", default=1)
    req_grp.add_argument('-index', '--index', type=str, help="index file", default = None)
    req_grp.add_argument('--model', type=str, help="Select training model.", required=True)
    req_grp.add_argument('--load', type=str, help="load model from file.", default=None)
    req_grp.add_argument('--trait', type=str, help="give trait a name.", default=None)
    req_grp.add_argument('-o', '--output', type=str, help="Input output dir.")
    req_grp.add_argument('-r', '--round', type=int, help="training round.", default=10)
    req_grp.add_argument('-lr', '--lr', type=float, help="Learning rate.", default=0.0001)
    req_grp.add_argument('-epo', '--epoch', type=int, help="training epoch.", default=50)
    req_grp.add_argument('-batch', '--batch', type=int, help="batch size.", default=16)
    req_grp.add_argument('--rank', type=bool, help="If the trait is a ranked value, will use a standard value instead.", default=False)
    req_grp.add_argument('-plot', '--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    req_grp.add_argument('-residual', '--residual', dest='residual', action='store_true')
    parser.set_defaults(residual=True)
    req_grp.add_argument('-quiet', '--quiet', type=int, help="silent mode, 0: quiet, 1: normal, 2: verbose", default=0)
    req_grp.add_argument('-save', '--save', type=bool, help="save model True/False",
                         default=False)
    req_grp.add_argument('-config', '--config', type=str, help='config file path, default: ./ML_composer.ini',
                         default="./ML_composer.ini")

    ### Neural model default attributes##
    req_grp.add_argument('--width', type=int, help="Hidden layer width (units).", default=8)
    req_grp.add_argument('--depth', type=int, help="Hidden layer depth.", default=4)
    parser.add_argument('--use-mean', dest='mean', action='store_true')
    parser.set_defaults(mean=False)

    args = parser.parse_args()

    return args


def plot_loss_history(h, title,plot_name=None,checkpoint=0):
    print("Plotting loss history...")
    plt.plot(h.history['loss'][5:], label = "Train loss", color = "blue")
    plt.plot(h.history['val_loss'][5:], label = "Validation loss", color = "red")
    plt.xlabel('Epochs')
    plt.title(title)
    #print plot name
    print("Plot name: ", plot_name)
    if plot_name and checkpoint == 0:
        #plt.legend()
        plt.savefig(plot_name)
        plt.close()
    else:
        plt.show()
    #plt.show()

class ML_composer:

    def __init__(self,silence=0):
        self._raw_data = {"GENO":pd.DataFrame(),"PHENO":pd.DataFrame(),"INDEX":pd.DataFrame(),"ANNOTATION":pd.DataFrame()}
        self.train_data = None
        self.train_info = pd.DataFrame()
        self.train_pheno = pd.DataFrame()
        self.valid_data = None
        self.valid_info = pd.DataFrame()
        self.valid_pheno = pd.DataFrame()
        self._model = {"INIT_MODEL":Model(args=None),"TRAINED_MODEL":Model(args=None)}
        self._info = {}
        self.method = None
        self.modelling = None
        self.silence_mode = silence
        self.config = None
        self.save = True
        self.plot = False
        self.args = None
        self.batchSize = 16
        self.record = pd.DataFrame(columns=["Trait", "TrainSet", "ValidSet", "Model", "Test_Accuracy",
                          "Valid_Accuracy", "MSE", "Runtime"])
        self.model_name = None

    def load_data(self,raw_data,raw_model,raw_info):
        #read outside dict of data
        self._raw_data = raw_data
        self._model = raw_model
        self._info = raw_info

        return

    def get_data(self,configer,args):
        self.args = args
        self.config = configer
        self._model = {"INIT_MODEL":Model(self.args),"TRAINED_MODEL":Model(self.args)}
        self._raw_data["GENO"] = pd.read_table(args.ped+".ped",sep="\t",header=None)
        self._raw_data["PHENO"] = pd.read_table(args.pheno, sep="\t", header=None)
        self._raw_data["INDEX"] = pd.read_table(args.index,sep="\t", header=None)
        self._info["CROSS_VALIDATE"] = sorted(self._raw_data["INDEX"].iloc[:,-1].unique())
        self.batchSize = args.batch
        print(self._raw_data["INDEX"].iloc[:,-1].value_counts().sort_values())

        self._raw_data["INFO"] = self._raw_data["GENO"].iloc[:,0:6]  #Further using fam file instead.

        print("Get genotype shape:",self._raw_data["GENO"].iloc[:,6:].shape)
        print(self._raw_data["GENO"].iloc[:,6:].iloc[1:10,1:10])
        self.plot = self.args.plot
        """
        #self.train_data = self._raw_data["GENO"].query('X.1 in @train_sample').query('X.1 not in @remove_list').iloc[:,6:]
        #self.valid_data = self._raw_data["GENO"].query('X.1 in @valid_sample').iloc[:,6:]
        #print(self.train_data.shape)
        #print(self.valid_data.shape)

        #self.train_data = filtered_data.query('Series in @train_year').query('Sample not in @remove_list').drop(["Series","Sample"], axis=1)
        #self.valid_data = filtered_data.query('Series in @valid_year').drop(["Series","Sample"], axis=1)
        """
        return

    def prepare_model(self):
        # create a Model object

        if self.args.load is not None:
            self._model["INIT_MODEL"].load_model(self.args.load)
        else:
            self._model["INIT_MODEL"].init_model()
        # save model summary to a txt file under the output directory
        self.model_name = self._model["INIT_MODEL"].get_model_name()

        # get data requirements - dimension, annotations, etc
        return

    def prepare_cross_validate(self):
        index_ref = []
        for idx in self._info["CROSS_VALIDATE"]:
            train_index = [x for x in self._info["CROSS_VALIDATE"] if x is not idx]
            valid_index = [idx]
            index_ref.append((train_index,valid_index))

        return index_ref

    def prepare_training(self,train_index:list,valid_index:list):
        
        removal = np.where(self._raw_data["PHENO"].iloc[:, int(self.args.mpheno)+1].isin([-9,None,"NA",np.nan]))[0].tolist()
        print("Overall population: {}".format(len(self._raw_data["INDEX"].index)))
        print("{} individuals need to be removed due to the miss phenotype".format(len(removal)))
        train_mask = [x for x in np.where(self._raw_data["INDEX"].iloc[:, -1].isin(train_index))[0].tolist() if x not in removal]
        valid_mask = [x for x in np.where(self._raw_data["INDEX"].iloc[:, -1].isin(valid_index))[0].tolist() if x not in removal]
        print("Filtered population: {}".format(len(train_mask)+len(valid_mask)))
        self.train_data = self._raw_data["GENO"].iloc[train_mask, 6:]
        self.valid_data = self._raw_data["GENO"].iloc[valid_mask, 6:]

        self.train_pheno = self._raw_data["PHENO"].iloc[train_mask,self.args.mpheno + 1]
        print("Mean of train phenotype:",np.mean(self.train_pheno))
        mean_train = np.mean(self.train_pheno)
        if self.args.mean is not True:
            print("Use raw phenotype as the target")
            mean_train = 0
        self.train_pheno = self.train_pheno - mean_train
        self.valid_pheno = self._raw_data["PHENO"].iloc[valid_mask, self.args.mpheno + 1]
        self.valid_pheno = self.valid_pheno - mean_train
        print(self.valid_pheno.head(5))

        #label_encoder = LabelEncoder()

        self.prepare_model()
        #self.train_data,self.train_pheno = self._model["INIT_MODEL"].data_transform(self.train_data,self.train_pheno, pheno_standard = self.args.rank) ## The raw data to transform include geno, pheno, annotations
        #self.valid_data,self.valid_pheno = self._model["INIT_MODEL"].data_transform(self.valid_data,self.valid_pheno, pheno_standard = self.args.rank)

        #self.train_data = np.asarray(self.train_data).astype(np.float32)
        self.train_pheno = np.asarray(self.train_pheno).astype(np.float32)
        #self.valid_data = np.asarray(self.valid_data).astype(np.float32)
        self.valid_pheno = np.asarray(self.valid_pheno).astype(np.float32)

        return

    def train(self,features_train, features_test, target_train, target_test,round=1):

        if type(features_train) is list:
            n_features = features_train[0].shape[1:]
        else:
            n_features = features_train.shape[1:]
        self._model["TRAINED_MODEL"] = self._model["INIT_MODEL"].modelling(
            input_shape = n_features,args = self.args, lr=float(self.args.lr))
        if round == 1:
            with open(os.path.abspath(self.args.output) + "/model_summary.txt", "w") as fh:
                self._model["TRAINED_MODEL"].summary(print_fn=lambda x: fh.write(x + "\n"))

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE)

        try:
            print(self._model["TRAINED_MODEL"].summary())
        except:
            print("It is a sklean-Random-forest model.")

        startTime = datetime.now()

        history = self._model["TRAINED_MODEL"].fit(
            features_train, target_train,
            epochs=int(self.args.epoch),
            validation_data=(features_test, target_test), verbose=int(self.args.quiet),
            callbacks=[callback],batch_size = self.batchSize)


        # let's just print the final loss
        print(' - train loss     : ' + str(history.history['loss'][-1]))
        print(' - validation loss: ' + str(history.history['val_loss'][-1]))
        print(' - loss decrease rate in last 5 epochs: ' + str(
            np.mean(np.gradient(history.history['val_loss'][-5:]))))
        print(' - Actual Training epochs: ', len(history.history['loss']))
        #print(self._model["TRAINED_MODEL"].predict(features_test).shape)
        test_length = target_test.shape[0]
        y_pred = np.reshape(self._model["TRAINED_MODEL"].predict(features_test), (test_length,))
        test_accuracy = np.corrcoef(y_pred, target_test)[0, 1]
        print("Train End.")
        print("In-year accuracy (measured as Pearson's correlation) is: ", test_accuracy)
        endTime = datetime.now()
        runtime = endTime - startTime
        print("Training Runtime: ", runtime.seconds / 60, " min")

        return history,test_accuracy,runtime

    def compose(self,train_index:list,valid_index:list,val=1):

        #features_train, features_test, target_train, target_test = train_test_split(self.train_data, self.train_pheno,test_size=0.2)
        features_train,target_train = self._model["INIT_MODEL"].data_transform(self.train_data,self.train_pheno, pheno_standard = self.args.rank)
        features_val,target_val = self._model["INIT_MODEL"].data_transform(self.valid_data,self.valid_pheno, pheno_standard = self.args.rank)

        print("Train status:")
        print("Epochs: ",self.args.epoch)
        print("Repeat(Round): ",self.args.round)
        #print("feature shape:",features_train.shape)

        round = 1
        while round <= self.args.round:
            history, test_accuracy, runtime = self.train(features_train, features_val, target_train, target_val,round=round)
            valid_accuracy, mse = self.model_validation()
            self.record.loc[len(self.record)] = [self.args.trait, train_index, valid_index, self.model_name,
                               test_accuracy, valid_accuracy, mse, runtime.seconds / 60]
            check_usage()
            if self.plot is True:
                # create a folder to save the plot, folder name: trait, model
                print("Plotting the training process...")
                plot_dir = os.path.abspath(self.args.output) + "/{}_{}_{}".format(self.args.trait, self.model_name,
                                                                                  self.args.trait)
                print(plot_dir)
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                # create a file name for the plot: path, model name, trait and round
                plot_name = plot_dir + "/{}_{}_{}.png".format(self.args.trait, self.model_name, val)
                # plot_name = os.path.abspath(self.args.output) + "/" + self.args.model + "_" + self.args.trait + "_" + str(round) + ".png"
                plot_loss_history(history, self.args.trait, plot_name,round-self.args.round)
            self._model["TRAINED_MODEL"] = None
            keras.backend.clear_session()
            gc.collect()
            round += 1
        if self.save == True:
            self.export_record()
        return

    def model_validation(self):

        valid_data,valid_pheno = self._model["INIT_MODEL"].data_transform(
            self.valid_data,self.valid_pheno, pheno_standard = self.args.rank)
        print("Predicting valid set..")
        val_length = valid_pheno.shape[0]
        y_pred_valid = np.reshape(self._model["TRAINED_MODEL"].predict(valid_data), (val_length,))
        print("Testing prediction:")
        print("Predicted: ", y_pred_valid[:10])
        print("observed: ", valid_pheno[:10])
        accuracy_valid = np.corrcoef(y_pred_valid, valid_pheno)[0, 1]
        mse = mean_squared_error(y_pred_valid, valid_pheno)

        print("Future prediction accuracy (measured as Pearson's correlation) is: ",
              accuracy_valid)
        return accuracy_valid,mse

    def export_record(self):

        self.record.to_csv("{}/{}_train_record_{}.csv".format(os.path.abspath(self.args.output), self.model_name, self.args.trait), sep="\t")
        print("Result:")
        print(self.record)

        return

class Model:

    def __init__(self,args):

        self.args = args
        self._init_model = NN(args)
        self._data_requirements = None
        self.modelling = None
        self.data_transform = None

    def get_model_name(self):
        return self._init_model.model_name()

    def init_model(self):

        self._init_model = MODELS[self.args.model](self.args)
        self.data_transform = self._init_model.data_transform
        self.modelling = self._init_model.model

        return

    def load_model(self,path):

        self._init_model = MODELS[self.args.model](self.args)
        self.data_transform = self._init_model.data_transform
        self.modelling = self._init_model.model
        self.modelling = keras.models.load_model(path)

        #self._init_model = load(path)
        return


def get_model_summary(model: tf.keras.Model) -> str:
    string_list = []
    model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)

def main():
    args = get_args()
    '''
    config_path = os.path.abspath(args.config)
    print("Get config file path from: ", config_path)
    config = configparser.ConfigParser()
    if platform.system().lower() == "windows":
        print(config_path)
        config.read(config_path)
    else:
        config.read(config_path)
    '''

    """
    Create folders from given output path
    """
    if args.output[0] == "/":
        locat = '/' + args.output.strip('/') + '/'
    else:
        locat = args.output.strip('/') + '/'
    if not os.path.exists(locat):
        os.mkdir(locat)

    ##write args to a txt file in the locat
    with open(locat + 'args.txt', 'w') as f:
        f.write(str(args))

    composer = ML_composer()
    composer.get_data(configer=None,args=args)
    composer.prepare_model()


    index_ref = composer.prepare_cross_validate()
    i = 1
    for train_idx,valid_idx in index_ref:
        print("Cross-validate: {}".format(i))
        composer.prepare_training(train_idx,valid_idx)
        composer.compose(train_idx,valid_idx,valid_idx[0])
        i+=1

if __name__ == "__main__":
    main()




