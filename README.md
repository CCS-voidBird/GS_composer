# GS_composer
GS_composer for Genomic prediction

This repositorie was a backup version for some publish reproduce needs; For other use, please email author (uqcche32@uq.edu.au) to get latest information.


Required: 
          A plink like ped file for genotypes - (numeric alleles for each SNP)
          A plink like phenotype file for phenotypes. format: FID,IID,Trait1... seperated by tabs       
          A Index file for cross-valiation. Format: FID, IID, Index

Current available/stable models: (Use Key as call parameter)
```
MODELS = {
    "MLP": MLP,
    "Numeric CNN": NCNN,
    "Binary CNN": BCNN,
    "MultiLevel Attention": MultiLevelAttention, #An in-house modified transformer model
} # All the models are stored in ClassModel.py
```
Example:
```
loss="r2"
LB=10
LC=16
AB=1
python ./GS_composer.py --build --analysis\
	--ped $geno --pheno $pheno --mpheno 1 --index $index --trait $trait \
	--model "MultiLevel Attention" --width 256 --depth 2 --addNorm\
	--locallyConnect $LC --embedding $LC --AttentionBlock $AB --batch 18 --num-heads 1 --locallyBlock $LB --epistatic \
	--epoch 30 --round 1 --lr 0.001 --loss $loss \
	-o $target --quiet 1 --plot \
	--save 
```

##Please use GS_composer as main py file for Deep learning related prediction, put rest of .py files under the same directory.
```
usage: GS_composer.py [-h] [--ped PED] [-pheno PHENO] [-mpheno MPHENO]
                      [-index INDEX] [-vindex VINDEX] [-annotation ANNOTATION]
                      [-o OUTPUT] [--trait TRAIT] [-build] [-analysis]
                      [--width WIDTH] [--depth DEPTH] [--use-mean]
                      [--model MODEL] [--load LOAD] [--data-type DATA_TYPE]
                      [-r ROUND] [-lr LR] [-epo EPOCH] [--num-heads NUM_HEADS]
                      [--activation ACTIVATION] [--embedding EMBEDDING]
                      [--locallyConnect LOCALLYCONNECT]
                      [--locallyBlock LOCALLYBLOCK]
                      [--AttentionBlock ATTENTIONBLOCK] [-batch BATCH]
                      [-loss LOSS] [--rank RANK] [-plot] [-epistatic]
                      [-addNorm] [-residual] [-quiet QUIET] [-save SAVE]
                      [-config CONFIG]

optional arguments:
  -h, --help            show this help message and exit

General:
  --ped PED             PED-like file name
  -pheno PHENO, --pheno PHENO
                        Phenotype file.
  -mpheno MPHENO, --mpheno MPHENO
                        Phenotype columns (start with 1).
  -index INDEX, --index INDEX
                        index file
  -vindex VINDEX, --vindex VINDEX
                        index for validate
  -annotation ANNOTATION, --annotation ANNOTATION
                        annotation file,1st row as colname
  -o OUTPUT, --output OUTPUT
                        Input output dir.
  --trait TRAIT         give trait a name.

Task Options:
  -build, --build       Full model process.
  -analysis, --analysis

Model Options:
  --width WIDTH         FC layer width (units).
  --depth DEPTH         FC layer depth.
  --use-mean
  --model MODEL         Select training model from MLP, Numeric CNN, Binary
                        CNN, Test CNN, Duo CNN, Double CNN, Attention CNN,
                        MultiHead Attention LNN, ResMLP, LNN, MultiLevel
                        Attention, MultiLevelNN.
  --load LOAD           load model from file.
  --data-type DATA_TYPE
                        Trait type (numerous, ordinal, binary)
  -r ROUND, --round ROUND
                        training round.
  -lr LR, --lr LR       Learning rate.
  -epo EPOCH, --epoch EPOCH
                        training epoch.
  --num-heads NUM_HEADS
                        (Only for multi-head attention) Number of heads,
                        currently only recommand 1 head.
  --activation ACTIVATION
                        Activation function for hidden Dense layer.
  --embedding EMBEDDING
                        (Only for multi-head attention) Embedding length
                        (default as 8)
  --locallyConnect LOCALLYCONNECT
                        (Only work with locally connected layers)
                        locallyConnect Channels (default as 1)
  --locallyBlock LOCALLYBLOCK
                        (Only work with locally connected layers) Length of
                        locallyBlock segment (default as 10)
  --AttentionBlock ATTENTIONBLOCK
                        (Only work with Attention layers) AttentionBlock
                        numbers (default as 1)
  -batch BATCH, --batch BATCH
                        batch size.
  -loss LOSS, --loss LOSS
                        loss founction.
  --rank RANK           If the trait is a ranked value, will use a standard
                        value instead.
  -plot, --plot
  -epistatic, --epistatic
  -addNorm, --addNorm
  -residual, --residual
  -quiet QUIET, --quiet QUIET
                        silent mode, 0: quiet, 1: normal, 2: verbose
  -save SAVE, --save SAVE
                        save model True/False
  -config CONFIG, --config CONFIG
                        config file path, default: ./ML_composer.ini

```

#Using backend GS_RF_composer as Random Forest related GP. Same parameters were inherited from the above scripts.

Example:
python $TMPDIR/ML_composer/GS_composer.py --ped $geno --pheno $pheno --mpheno 1 --index $index --vindex 1 --trait smut --leave $leave --tree $tree --model "Random Forest" -o ./Random_forest --quiet 1 --leave 50 100 --tree 100 200

```
usage: GS_RF_composer.py [-h] --ped PED -pheno PHENO [-mpheno MPHENO] [-index INDEX] [-vindex VINDEX] --model MODEL [--load LOAD] [--trait TRAIT] [-o OUTPUT] [-r ROUND] [--rank RANK] [-plot PLOT] [-sli SILENCE]
                         [-save SAVE] [-config CONFIG] [--leave LEAVE [LEAVE ...]] [--tree TREE [TREE ...]]

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
  -vindex VINDEX, --vindex VINDEX
                        index for validate
  --model MODEL         Select training model.
  --load LOAD           load model from file.
  --trait TRAIT         give trait a name.
  -o OUTPUT, --output OUTPUT
                        Input output dir.
  -r ROUND, --round ROUND
                        training round.
  --rank RANK           If the trait is a ranked value, will use a standard value instead.
  -plot PLOT, --plot PLOT
                        show plot?
  -sli SILENCE, --silence SILENCE
                        silent mode
  -save SAVE, --save SAVE
                        save model True/False
  -config CONFIG, --config CONFIG
                        config file path, default: ./ML_composer.ini
  --leave LEAVE [LEAVE ...]
                        tree leaf options.
  --tree TREE [TREE ...]
                        tree population options.

```
