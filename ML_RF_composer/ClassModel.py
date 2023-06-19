from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import numpy as np
import configparser
from Functions import *

####################
class NN():

    def __init__(self):
        self.name = "NN"

    def model_name(self):
        # get class name
        return self.__class__.__name__

    def data_transform(self, geno, pheno, anno=None, pheno_standard=False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno, pheno

    def model(self):
        pass


class RF():

    def __init__(self):
        self.name = "Random Forest"

    def model_name(self):
        # get class name
        return self.__class__.__name__

    def data_transform(self, geno, pheno, anno=None, pheno_standard=False):
        print("USE Numeric encoding")
        geno = decoding(geno)
        # geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno, pheno

    def model(self,args):
        model = RandomForestRegressor(n_jobs=-1, random_state=0,oob_score=False, verbose=1,
                                      max_features=args["leaves"],
                                      n_estimators=args["trees"])
        return model


def RF_backup(config=None, specific=False, n_features=500, n_estimators=200):
    if specific == True:
        model = RandomForestRegressor(n_jobs=-1, random_state=0, criterion="mse", oob_score=False, verbose=1,
                                      max_features=n_features,
                                      n_estimators=n_estimators)
        return model
    else:
        rm_config = {x: int(config["RM"][x]) for x in config["RM"].keys()}
        try:
            model = RandomForestRegressor(n_jobs=-1, random_state=0,oob_score=False, verbose=1,
                                          **rm_config)
        except:
            print("Cannot find the config file")
            model = RandomForestRegressor(n_jobs=-1, random_state=0, criterion="mse", oob_score=False, verbose=1,
                                          n_estimators=2000)

        return model


MODELS = {
    "Random Forest": RF,
}


def main():
    print("Main function from ClassModel.py")
    # tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)


if __name__ == "__main__":
    main()
