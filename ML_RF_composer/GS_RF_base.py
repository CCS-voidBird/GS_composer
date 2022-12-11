
from Functions import *
from GSModel import *
import pandas as pd
import matplotlib.pyplot as plt
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

##############################################################
##########Training requirement################################
##Training by para-sets ###
##GOING to Using a PLINK like format###
##Output format: an table with mean accuracy for each para set; A density plot for each accuracy##########
#################################################################

CNNs = ["CNN","TDCNN","DeepGS"]

"""
GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"

TRAIN_PATH = "E:/learning resource/PhD/sugarcane/2015_TCHBlup_2000.csv"
VALID_PATH = "E:/learning resource/PhD/sugarcane/2016_TCHBlup_2000.csv"
"""

def plot_loss_history(h, title):
    plt.plot(h.history['loss'], label = "Train loss")
    plt.plot(h.history['val_loss'], label = "Validation loss")
    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    #plt.show()

class ML_composer:

    def __init__(self,silence=0):
        self.train_data = None
        self.valid_data = None
        self.method = None
        self.modelling = None
        self.silence_mode = silence
        self.sub_selection = None
        self.keeping = None
        self.traits = None
        self.records = None
        self.subset_index = []
        self.config = None
        self.save = True
        self.plot = False

    def get_data(self, configer):
        self.config = configer
        self.method = configer["BASIC"]["method"]
        self.silence_mode = configer["BASIC"]["silence"]
        self.modelling = METHODS[self.method]
        self.traits = configer["BASIC"]["traits"].split("#")
        self.sub_selection = configer["BASIC"]["sub_selection"]
        self.keeping = configer["BASIC"]["non_genetic_factors"].split("#")
        if "None" in self.keeping: self.keeping = []
        train_year = get_years(configer["BASIC"]["train"])
        valid_year = get_years(configer["BASIC"]["valid"])

        print("Getting data from config file path..")

        try:
            geno_data = pd.read_csv(self.config["PATH"]["genotype"],
                                    sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
            pheno_data = pd.read_csv(self.config["PATH"]["phenotype"],
                                     sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
            blues = pd.read_csv("~/Blue_phenotypes.txt", sep="\t")
        except:
            try:
                print("Using backup path (for trouble shooting)")
                print(self.config["BACKUP_PATH"]["genotype"])
                geno_data = pd.read_csv(self.config["BACKUP_PATH"]["genotype"],
                                        sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
                pheno_data = pd.read_csv(self.config["BACKUP_PATH"]["phenotype"],
                                         sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
            except:
                print("No valid path found.")
                exit()

        train_sample = pheno_data.query(
            "Series in @train_year"
        ).Clone.unique()

        valid_sample = pheno_data.query(
            "Series in @valid_year"
        ).Clone.unique()

        print("Train clones: {}, valid clones: {}".format(len(train_sample), len(valid_sample)))

        # The default column name of individual is "Clone"

        non_genetic_factors = [x for x in pheno_data.columns if x not in self.traits and x not in ["Sample", "Clone"]]
        print("Detected non-genetic factors from phenotype file: ", non_genetic_factors)

        geno_data = decoding(geno_data)
        if self.config["BASIC"]["use_blue"] == '1':
            print("Use BLUE value as phenotypes")
            filtered_data = read_pipes(geno_data, pheno_data, train_year + valid_year, blue=blues)
        else:
            filtered_data = read_pipes(geno_data, pheno_data, train_year + valid_year)
        if self.config["BASIC"]["sub_selection"] == '1':
            print("The sub_selection is enabled, thus switch to pure genetic prediction mode.")
            self.subset_index = pheno_data.Region.unique()  # Record subselection index (e.g. for region:N, B..
        else:
            print("The sub_selection is disabled..")
            print("Below factors will be fed into models: ")
            print(self.keeping)  # Useful non_genetic factors e.g.   Series, Region and other..

        remove_list = []
        if self.config["BASIC"]["Strict"] == "1":
            print("Strict training model detected, remove overlapping in training data..")
            print("Preview of data columns:")
            print(filtered_data.columns)
            remove_list = np.intersect1d(train_sample, valid_sample)
            print("Finished")

        dropout = [x for x in non_genetic_factors if
                   x not in self.keeping]
        print("Removing useless non-genetic factors: {}".format(dropout))

        try:
            filtered_data.drop(dropout, axis=1, inplace=True)
        except:
            print(dropout, "already removed.")

        print(filtered_data.columns[1:10])
        print("Get data shape:", filtered_data.shape)
        print(filtered_data.iloc[1:10, 1:10])

        self.train_data = filtered_data.query('Sample in @train_sample').query('Sample not in @remove_list').drop(
            ["Sample", "Clone"], axis=1)
        self.valid_data = filtered_data.query('Sample in @valid_sample').drop(["Sample", "Clone"], axis=1)
        print(self.train_data.shape)
        print(self.valid_data.shape)

        # self.train_data = filtered_data.query('Series in @train_year').query('Sample not in @remove_list').drop(["Series","Sample"], axis=1)
        # self.valid_data = filtered_data.query('Series in @valid_year').drop(["Series","Sample"], axis=1)

        return

    def prepare_training(self,trait,other_factor = 'Region',factor_value = 'all',test=True):

        if self.config["BASIC"]["sub_selection"] == '1' and factor_value != 'all':
            print("Creating subsets by non_genetic_factors..")
            print("The reference factor index: {} in {}".format(factor_value,other_factor))
            in_train = self.train_data.dropna(subset=[trait], axis=0).query('Region == @factor_value').drop(
                self.keeping,axis=1
            )
            in_valid = self.valid_data.dropna(subset=[trait], axis=0).query('Region == @factor_value').drop(
                self.keeping,axis=1
            )

        else:
            print("Use the default setting, add non-genetic factors as attributes")
            in_train = self.train_data.dropna(subset=[trait], axis=0)
            in_valid = self.valid_data.dropna(subset=[trait], axis=0)

        label_encoder = LabelEncoder()
        train_targets = in_train[trait].values  # Get the target values from train set
        valid_targets = in_valid[trait].values
        train_features = in_train.drop(self.traits, axis=1)
        valid_features = in_valid.drop(self.traits, axis=1)
        print("The subset index contains: ", self.subset_index)
        print("currently the training method is: ", self.method)
        if self.method == "RF":
            print("Currently only support RF training with numeric values, thus, the data will exclude non-genetic factos.")
            try:
                print("Removing non-genetic factors: ",self.keeping)
                train_features.drop(self.keeping,axis=1,inplace=True)
                valid_features.drop(self.keeping,axis=1,inplace=True)
            except:
                print("These non_genetic factors are already removed: ",self.keeping)
                print(train_features.columns)

            print("The selected region is: ",factor_value)

        return train_features,train_targets,valid_features,valid_targets

    def make_hp_set(self):
        hps = [hp for hp in self.config[self.method]]
        hp_sets = list(combi([v.split(",") for k,v in dict(self.config[self.method]).items()]))
        print("The hyper-parameter is: \n","\t".join(hps))
        for content in hp_sets:
            print("\t".join(content))
        return hps,hp_sets

    def make_forest(self,model_path,record_path):
        hps, hp_sets = self.make_hp_set()
        record_cols = ["trait", "trainSet", "validSet"]+hps+["test_score", "valid_score", "test_accuracy","accuracy", "mse","Region", "Runtime"]
        accs = []  # pd.DataFrame(columns=["trait","trainSet","validSet","score","cov"])
        records = []
        paraList = []
        
        #max_feature_list = [int(x) for x in self.config["RM"]["max_features"].split(",")]
        """
        A tuple of RF model hp: 1st: max_features, 2nd: n_trees, 3..4.. 
        """
        print("The subset index contains: ",self.subset_index)
        print(self.subset_index+["all"])
        for trait in self.traits:
            for region in self.subset_index+["all"]:
                print(trait)
                avg_acc = []
                avg_score = []
                acg_same_score = []
                avg_mse = []
                avg_runtime = []
                avg_test_acc = []
                best_model = [0,None]
                for hp_set in hp_sets:
                    print("Modified hyper-parameter names: \n", "\t".join(hps))
                    print("Now training by the following hyper_parameters: ".format(hp_set))
                    print("Now training by region: {}".format(region))
                    r = 0
                    while r < int(self.config["BASIC"]["replicate"]):
                        startTime = datetime.now()

                        train_features, train_targets, valid_features, valid_targets = self.prepare_training(trait,
                                                                                                             factor_value=region)

                        features_train, features_test, target_train, target_test = train_test_split(train_features,
                                                                                                  train_targets,
                                                                                                  test_size=0.2)

                        """'
                        # testing set spilt part particularly for CNN model
                        features_train_val, features_val_val, target_train_val, target_val_val = train_test_split(
                            features_test,
                            target_test,
                            test_size=0.5)
                        print(features_train)
                        """
                        model = RF(specific=True,n_estimators=int(hp_set[0]), n_features=int(hp_set[1]))

                        model.fit(features_train, target_train)
                        endTime = datetime.now()
                        runtime = endTime - startTime
                        print("Runtime: ", runtime.seconds / 60, " min")
                        same_score = model.score(features_test, target_test)  # Calculating accuracy in the same year
                        score = model.score(valid_features, valid_targets)

                        test_predict = model.predict(features_test) # Testing at the end of the training
                        test_accuracy = np.corrcoef(test_predict, target_test)[0, 1]

                        n_predict = model.predict(valid_features) # Testing with the valid set - outside the training
                        obversed = np.squeeze(valid_targets)

                        accuracy = np.corrcoef(n_predict, obversed)[0, 1]
                        mse = mean_squared_error(obversed, n_predict)
                        print("The accuracy for {} in RM is: {}".format(trait, accuracy))
                        print("The mse for {} in RM is: {}".format(trait, mse))
                        print("The variance for predicted {} is: ".format(trait, np.var(n_predict)))
                        """
                        # Looking at the distribution of the predicted values
                        print("A bite of output:")
                        print("observe: ", obversed[:50])
                        print("predicted: ", n_predict[:50])
                        
                        save_df = pd.DataFrame({"obv": obversed, "pred": n_predict})
                        save_df.to_csv("~/saved_outcomes.csv", sep="\t")
                        """
                        avg_acc.append(accuracy)
                        avg_score.append(score)
                        acg_same_score.append(same_score)
                        avg_mse.append(mse)
                        avg_runtime.append(runtime)
                        avg_test_acc.append(test_accuracy)
                        if accuracy > best_model[0]:
                            best_model = [accuracy, model]
                        r += 1
                        records.append([trait, "2013-15", "2017", hp_set[0],hp_set[1], same_score, score,test_accuracy,accuracy, mse, region,runtime.seconds / 60])
                    if self.save is True:
                        saved_rf_model_fn = "{}{}_{}_{}_model.json".format(model_path, trait, self.method, region)
                        pickle.dump(best_model[1], open(saved_rf_model_fn, "wb"))
                    accs.append([trait, "2013-15", "2017", hp_set[0],hp_set[1], np.mean(acg_same_score), np.mean(avg_score),np.mean(avg_test_acc),
                                 np.mean(avg_acc), np.mean(avg_mse),region,np.mean(avg_runtime).seconds / 60])
                    record_train_results(records, record_cols, method=self.method, path=record_path)
            check_usage()
            record_train_results(records, record_cols, method=self.method, path=record_path)
            record_train_results(accs, record_cols, self.method, path=record_path, extra="_summary")

        #record_train_results(accs, cols=record_cols, method="RM", path="~", para="max_features")
        #record_train_results(records, cols=record_cols, method="RM", path="~", para="max_feature_raw")



def main():

    args = get_args()
    config_path = os.path.abspath(args.config)
    print("Get config file path from: ",config_path)
    config = configparser.ConfigParser()
    if platform.system().lower() == "windows":
        print(config_path)
        config.read(config_path)
        #print(config["BACKUP_PATH"]["genotype"])
    else:
        config.read(config_path)

    """
    Create folders from given output path
    """
    if config["OUTPUT"]["output"][0] == "/":
        locat = '/' + config["OUTPUT"]["output"].strip('/') + '/'
    else:
        locat = config["OUTPUT"]["output"].strip('/') + '/'
    if not os.path.exists(locat):
        os.mkdir(locat)
    locat = locat + "{}_vs_{}/".format(config["BASIC"]["train"], config["BASIC"]["valid"])
    model_path = locat + "models/"
    record_path = locat + "records_{}/".format(args.optimizer)
    for path in [locat,model_path,record_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    composer = ML_composer()
    composer.get_data(config)
    if config["BASIC"]["method"] in CNNs:
        print("Use GS_base_CNN.py instead.")
    elif config["BASIC"]["method"] == "RF":
        composer.make_forest(model_path=model_path,record_path=record_path)

if __name__ == "__main__":
    main()
