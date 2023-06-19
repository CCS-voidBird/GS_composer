library(dplyr)
#setwd("E:/learning resource/PhD/genomic data/Sugarcane/")
genos = read.csv("./qc_genotypes_decode.csv",sep="\t")
phenos = read.csv("./blups13_17.txt",sep="\t")
train_year = c("2013","2014","2015")
valid_year = c("2017")
train_set = phenos[phenos$Series %in% train_year,]
valid_set = phenos[phenos$Series %in% valid_year,]
whole_set = phenos[phenos$Series %in% c("2016","2013","2014","2015","2017"),]
samples = unique(phenos$Clone)
length(samples)

#phenos[which(phenos$TCHBlup == NA & phenos$CCSBlup == NA & phenos$FibreBlup == NA),]
dim(genos)
genotypes = genos[which(genos$Sample %in% whole_set$Clone),]

library(asreml)

TCH = whole_set$TCHBlup
CCS = whole_set$CCSBlup
Fibre = whole_set$FibreBlup
#setwd("E:/learning resource/OneDrive - The University of Queensland/Sugarcane")


model = asreml(fixed=TCHBlup ~ Series + Region + Trial + Crop +Clone,
               workspace=128e06,na.action = na.method(y="include"),
               data= whole_set)

TCHBlue <- summary(model,coef=T)$coef.fixed[1:dim(genotypes)[1],]

model = asreml(fixed=CCSBlup ~ Series + Region + Trial + Crop +Clone,
               workspace=128e06,na.action = na.method(y="include"),
               data= whole_set)

CCSBlue <- summary(model,coef=T)$coef.fixed[1:dim(genotypes)[1],]


model = asreml(fixed=FibreBlup ~ Series + Region + Trial + Crop +Clone,
               workspace=128e06,na.action = na.method(y="include"),
               data= whole_set)

FibreBlue <- summary(model,coef=T)$coef.fixed[1:dim(genotypes)[1],]

index = unlist(strsplit(rownames(TCHBlue),split = "_"))
index = index[!index=="Clone"]

finalBLUP = data.frame(Clone = index, TCHBlup = TCHBlue[,1], CCSBlup = CCSBlue[,1], FibreBlup = FibreBlue[,1])

write.table(finalBLUP,file="./Blue_phenotypes.txt",quote = F,row.names = F,sep="\t")

reference = read.csv("./phenotypes.csv",sep="\t")
phenos = read.table("./Blue_phenotypes.txt",sep="\t",h=T)

train_clones <- unique(reference[which(reference$Series %in% c("2013","2014","2015","2017")),]$Clone)
#train_clones <- unique(reference[which(reference$Series %in% c("2013","2014","2015","2016","2017")),]$Clone)
train_blues = phenos[which(phenos$Clone %in% train_clones),]
train_clones = data.frame(train_clones)
colnames(train_clones)[1] = "Clone"
train_blues = left_join(train_clones,train_blues,by="Clone") #cbind(train_blues$Clone,train_blues)
valid_clones = unique(reference[which(reference$Series == 2017),]$Clone)
train_blues = cbind(train_blues$Clone,train_blues)
train_blues[which(train_blues$Clone %in% valid_clones),][,3:5] = NA
write.table(train_blues,"./Blue_phenotypes_train_15.txt",sep="\t",row.names = F,col.names = F,quote = F)

train_clones <- unique(reference[which(reference$Series %in% c("2013","2014","2015","2016","2017")),]$Clone)
train_blues = phenos[which(phenos$Clone %in% train_clones),]
train_clones = data.frame(train_clones)
colnames(train_clones)[1] = "Clone"
train_blues = left_join(train_clones,train_blues,by="Clone") #cbind(train_blues$Clone,train_blues)
valid_clones = unique(reference[which(reference$Series == 2017),]$Clone)
train_blues = cbind(train_blues$Clone,train_blues)
train_blues[which(train_blues$Clone %in% valid_clones),][,3:5] = NA
write.table(train_blues,"./Blue_phenotypes_train_16.txt",sep="\t",row.names = F,col.names = F,quote = F)
