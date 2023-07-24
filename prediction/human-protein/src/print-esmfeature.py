import csv
from getdataset import getdataset_iFeature, getdataset_dscript, getdataset_esmFeature, printesmFeature
from votemodel import randomforest_esm_get_voting
import torch
import numpy as np
import getopt, sys

# ###############################################################    

if  __name__ == "__main__":

    proteinlist_file = "feature.list"

    esmfeature_type = "all"
   
    esmfeature_datapath = "~/preprotac/prediction/human-protein/data/"

    printesm_datapath = "~/preprotac/prediction/human-protein/pt_data/"

    printesmFeature(proteinlist_file,esmfeature_type,printesm_datapath)


