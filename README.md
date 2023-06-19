# GS_composer
GS_composer for Genomic prediction

Required: A plink like ped file for genotypes - (numeric alleles for each SNP);
          A plink like phenotype file for phenotypes. format: FID,IID,Trait1... seperated by tabs
          A Index file for cross-valiation. Format: FID, IID, Index

Current available models: (Use Key as call parameter)

MODELS = {
    "MLP": MLP, (Multilayer Perceptron)
    "Numeric CNN": NCNN,
    "Binary CNN": BCNN,
    "Test CNN":Transformer, (Need work)
    "Duo CNN": DCNN, Double portal CNN (Combined BCNN and NCNN)
    "Double CNN": DoubleCNN,
    "Attention CNN": AttentionCNN, (Need work)
    "ResMLP": ResMLP,
    "LNN": LNN, (Local connected Network)
}

Example:
python $TMPDIR/ML_composer/GS_composer.py --ped $geno --pheno $pheno --mpheno 1 --index $index --trait smut --width $width --depth $depth --model "Attention CNN" -o ./Attention_CNN_elu --quiet 1 --plot --residual

##Please use GS_composer as main py file for Deep learning related prediction
Usage:
usage: GS_composer.py [-h] --ped PED -pheno PHENO [-mpheno MPHENO]
                      [-index INDEX] --model MODEL [--load LOAD]
                      [--trait TRAIT] [-o OUTPUT] [-r ROUND] [-lr LR]
                      [-epo EPOCH] [-batch BATCH] [--rank RANK] [-plot]
                      [-residual] [-quiet QUIET] [-save SAVE] [-config CONFIG]
                      [--width WIDTH] [--depth DEPTH] [--use-mean]

optional arguments:
  -h, --help            show this help message and exit
  --use-mean

Required:
  --ped PED             PED-like file name
  -pheno PHENO, --pheno PHENO
                        Phenotype file.
  -mpheno MPHENO, --mpheno MPHENO
                        Phenotype columns (start with 1).
  -index INDEX, --index INDEX
                        index file
  --model MODEL         Select training model.
  --load LOAD           load model from file.
  --trait TRAIT         give trait a name.
  -o OUTPUT, --output OUTPUT
                        Input output dir.
  -r ROUND, --round ROUND
                        training round.
  -lr LR, --lr LR       Learning rate.
  -epo EPOCH, --epoch EPOCH
                        training epoch.
  -batch BATCH, --batch BATCH
                        batch size.
  --rank RANK           If the trait is a ranked value, will use a standard
                        value instead.
  -plot, --plot
  -residual, --residual
  -quiet QUIET, --quiet QUIET
                        silent mode, 0: quiet, 1: normal, 2: verbose
  -save SAVE, --save SAVE
                        save model True/False
  -config CONFIG, --config CONFIG
                        config file path, default: ./ML_composer.ini
  --width WIDTH         Hidden layer width (units).
  --depth DEPTH         Hidden layer depth.


#Using backend GS_RF_composer as Random Forest related GP.
usage: GS_RF_composer.py [-h] --ped PED -pheno PHENO [-mpheno MPHENO]
                         [-index INDEX] --model MODEL [--load LOAD]
                         [--trait TRAIT] [-o OUTPUT] [-r ROUND] [-lr LR]
                         [-epo EPOCH] [--rank RANK] [-plot PLOT]
                         [-sli SILENCE] [-save SAVE] [-config CONFIG]
                         [--width WIDTH] [--depth DEPTH]
                         [--leave LEAVE [LEAVE ...]] [--tree TREE [TREE ...]]

optional arguments:
  -h, --help            show this help message and exit

Required:
  --ped PED             PED-like file name
  -pheno PHENO, --pheno PHENO
                        Phenotype file.
  -mpheno MPHENO, --mpheno MPHENO
                        Phenotype columns (start with 1).
  -index INDEX, --index INDEX
                        index file
  --model MODEL         Select training model.
  --load LOAD           load model from file.
  --trait TRAIT         give trait a name.
  -o OUTPUT, --output OUTPUT
                        Input output dir.
  -r ROUND, --round ROUND
                        training round.
  -lr LR, --lr LR       Learning rate.
  -epo EPOCH, --epoch EPOCH
                        training epoch.
  --rank RANK           If the trait is a ranked value, will use a standard
                        value instead.
  -plot PLOT, --plot PLOT
                        show plot?
  -sli SILENCE, --silence SILENCE
                        silent mode
  -save SAVE, --save SAVE
                        save model True/False
  -config CONFIG, --config CONFIG
                        config file path, default: ./ML_composer.ini
  --width WIDTH         Hidden layer width (units).
  --depth DEPTH         Hidden layer depth.
  --leave LEAVE [LEAVE ...]
                        tree leaf options.
  --tree TREE [TREE ...]
                        tree population options.

