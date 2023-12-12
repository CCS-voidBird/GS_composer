from tensorflow import keras
import tensorflow as tf
from CustomLayers import *
from GS_composer import *
from Functions import *
from ClassModel import *
from keras import backend as K
import pandas as pd

################################################
###############Identify path################
################################################

#user_profile = "E:/learning resource/"
"""
user_profile = "H:/ML_archive/"
#assume model index is 1
model_index = 1
trait = "pachy"
model_series = "PIP_attention/MultiLevel/v"+str(model_index)+"_1AB_Epi_1000SNP_leaky_reluLinear"

################################################################
#####First stage: Input trained model from directory#########
################################################################

model_path = user_profile + model_series

model_name = "MultiLevelAttention_v"+str(model_index)+"_1AB_Epi_1000SNP_leaky_reluLinear"
model_full_path = model_path + "/" + str(trait) + "_MultiLevelAttention_"+str(model_index)

print(model_full_path)
model = keras.models.load_model(model_full_path)
model.summary()
#model = keras.Model(inputs=input1, outputs=QV_output)
model.compile(optimizer="RMSprop", loss="mean_squared_error")
"""

################################################################
#####Second stage: Create investigate dataset #########
################################################################


marker_dim = 1000

"""
1. Create three backgrounds


bg0 = np.zeros((1, marker_dim,1))
bg1 = np.ones((1, marker_dim,1))
# 1000 markers with value 2
bg2 = np.ones((1, marker_dim,1)) + 1
"""


"""
2. Create marker mutation sets


dataset = []

for bg in range(0,3):
    print("Background: {}".format(bg))
    large_matrix = []
    for allele in range(0,3):
        allele_matrix = np.full((marker_dim, marker_dim),bg,dtype=np.float32)
        np.fill_diagonal(allele_matrix,allele)
        np.expand_dims(allele_matrix,axis=-1)
        large_matrix.append(allele_matrix)
    dataset.append(large_matrix)

    
    #print(large_matrix.shape)
"""

################################################################
#####Third stage: Predict phenotypes by trained model #########
################################################################
"""
marker_contributs = []

for bg in range(0,3):
    print("Background: {}".format(bg))
    for dose in range(0,3):
        print("Analysing dose: {}".format(dose))
        gebvs = model.predict(dataset[bg][dose])
        bs = gebvs - bps[bg]
        bs = [bg,dose]+np.transpose(bs).tolist()[0]
        marker_contributs.append(bs)

marker_contributs = pd.DataFrame(marker_contributs)
print(marker_contributs.shape)
marker_contributs.columns = ["Background","Dose"]+["SNP_"+str(i) for i in range(1,marker_dim+1)]
marker_contributs.to_csv(model_path+"/marker_contributs.csv",index=False)
"""

###################################
#Placeholder for custmized function that investigate trained models##########
#####################################################################

def investigate_model(model=None,model_path=None,marker_dim=1000):

    if model is None:
        model = keras.models.load_model(model_path)
        model.compile(optimizer="RMSprop", loss="mean_squared_error")

    else:
        model = model
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        #model.compile(optimizer="RMSprop", loss="mean_squared_error")

    marker_contributs = []
    print(model.layers[0].input_shape)
    marker_dim = model.layers[0].input_shape[-1][-2]

    bg0 = np.zeros((1, marker_dim,1))
    bg1 = np.ones((1, marker_dim,1))
    bg2 = np.ones((1, marker_dim,1)) + 1

    bp0 = model.predict(bg0)
    bp1 = model.predict(bg1)
    bp2 = model.predict(bg2)
    bps = [bp0,bp1,bp2]
    print((bp0,bp2,bp1))

    dataset = []

    for bg in range(0,3):
        print("Now creating simulated marker set under Background: {}".format(bg))
        large_matrix = []
        for allele in range(0,3):
            allele_matrix = np.full((marker_dim, marker_dim),bg,dtype=np.float32)
            np.fill_diagonal(allele_matrix,allele)
            np.expand_dims(allele_matrix,axis=-1)
            large_matrix.append(allele_matrix)
        dataset.append(large_matrix)

    for bg in range(0,3):
        print("Now estimating marker effects under Background: {}".format(bg))
        for dose in range(0,3):
            print("Analysing dose: {}".format(dose))
            gebvs = model.predict(dataset[bg][dose])
            bs = gebvs - bps[bg]
            bs = [bg,dose]+np.transpose(bs).tolist()[0]
            marker_contributs.append(bs)

    marker_contributs = pd.DataFrame(marker_contributs)
    print(marker_contributs.shape)
    marker_contributs.columns = ["Background","Dose"]+["SNP_"+str(i) for i in range(1,marker_dim+1)]
    marker_contributs.to_csv(model_path+"/marker_contributes.csv",index=False,sep="\t")
    return 

if __name__ == "__main__":
    #user_profile = "E:/learning resource/"

    user_profile = "H:/ML_archive/"
    for i in range(1,6):
        model_index = i
        for trait in ["smut","pachy"]:

            model_series = "PIP_attention/MultiLevel/v"+str(model_index)+"_1AB_Epi_1000SNP_leaky_reluLinear"


            model_path = user_profile + model_series

            model_name = "MultiLevelAttention_v"+str(model_index)+"_1AB_Epi_1000SNP_leaky_reluLinear"
            model_full_path = model_path + "/" + str(trait) + "_MultiLevelAttention_"+str(model_index)
            investigate_model(model_path=model_full_path,marker_dim=marker_dim)