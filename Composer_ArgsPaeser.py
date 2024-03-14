import argparse
from ClassModel import MODELS, loss_fn

def get_args():
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group(title='General')
    general.add_argument('--ped', type=str, help="PED-like file name")
    general.add_argument('-pheno', '--pheno', type=str, help="Phenotype file.")
    general.add_argument('-mpheno', '--mpheno', type=int, help="Phenotype columns (start with 1).", default=1)
    general.add_argument('-index', '--index', type=str, help="index file", default = None)
    general.add_argument('-vindex', '--vindex', type=int, help="index for validate", default = None)
    general.add_argument('-annotation', '--annotation', type=str, help="annotation file,1st row as colname", default=None)
    general.add_argument('-o', '--output', type=str, help="Input output dir.",default="./Composed")
    general.add_argument('--trait', type=str, help="give trait a name.", default=None)

    task_opts = parser.add_argument_group(title='Task Options')
    task_opts.add_argument('-build', "--build", help="Full model process.", dest='build', action='store_true')
    parser.set_defaults(build=False)
    task_opts.add_argument('-analysis', '--analysis', help="Analysis only process.", dest='analysis', action='store_true')
    parser.set_defaults(analysis=False)
    
    build_args = parser.add_argument_group(title='Model Options')
    build_args.add_argument('--model', type=str, help="Select training model from {}.".format(", ".join(MODELS.keys())))

    ### Neural model default attributes##
    build_args.add_argument('--width', type=int, help="FC layer width (units).", default=8)
    build_args.add_argument('--depth', type=int, help="FC layer depth.", default=4)
    build_args.add_argument('--load', type=str, help="load model from file.", default=None)
    build_args.add_argument('--data-type', type=str, help="Trait type (numerous, ordinal, binary)", default="numerous")
    build_args.add_argument('-r', '--round', type=int, help="training round.", default=10)
    build_args.add_argument('-lr', '--lr', type=float, help="Learning rate.", default=0.0001)
    build_args.add_argument('-epo', '--epoch', type=int, help="training epoch.", default=50)
    build_args.add_argument('--num-heads', type=int, help="(Only for multi-head attention) Number of heads, currently only recommand 1 head.", default=1)
    build_args.add_argument('--activation', type=str, help="Activation function for hidden Dense layer.", default='relu')
    build_args.add_argument('--embedding', type=int, help="(Only for multi-head attention) Embedding length (default as 8)", default=8)
    build_args.add_argument('--locallyConnect', type=int, help="(Only work with locally connected layers) locallyConnect Channels (default as 1)", default=1)
    build_args.add_argument('--locallyBlock', type=int, help="(Only work with locally connected layers) Length of locallyBlock segment (default as 10)", default=10)
    build_args.add_argument('--AttentionBlock', type=int, help="(Only work with Attention layers) AttentionBlock numbers (default as 1)", default=1)
    build_args.add_argument('-batch', '--batch', type=int, help="batch size.", default=16)
    build_args.add_argument('--loss', type=str, help="loss founction from {}.".format(", ".join(loss_fn.keys())), default="mse")
    build_args.add_argument('--rank', type=bool, help="If the trait is a ranked value, will use a standard value instead.", default=False)
    build_args.add_argument('-quiet', '--quiet', type=int, help="silent mode, 0: quiet, 1: normal, 2: verbose", default=2)


    build_args.add_argument('-plot', '--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)

    build_args.add_argument('-epistatic', '--epistatic', dest='epistatic', action='store_true')
    parser.set_defaults(epistatic=False)

    build_args.add_argument('-addNorm', '--addNorm', dest='addNorm', action='store_true')
    parser.set_defaults(addNorm=False)

    build_args.add_argument('-maf', '--maf', help="Enable minor allele frequency multiplier, it will adjust genotype alleles with its MAF.", dest='maf', action='store_true')
    parser.set_defaults(maf=False)

    build_args.add_argument('-residual', '--residual', dest='residual', action='store_true')
    parser.set_defaults(residual=False)

    build_args.add_argument('-save', '--save', dest='save', action='store_true', help="save model True/False")
    parser.set_defaults(save=False)
    
    build_args.add_argument('-config', '--config', type=str, help='config file path, default: ./ML_composer.ini (Currently not available))',
                         default="./ML_composer.ini")

    build_args.add_argument('--use-mean', dest='mean', action='store_true')
    parser.set_defaults(mean=False)

    args = parser.parse_args()

    return args