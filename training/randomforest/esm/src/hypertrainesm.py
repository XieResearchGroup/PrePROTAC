import csv
from gridsearchrf import randomforest_gridsearch_ROC_train
from getdataset import getdataset_esmFeature, printesmFeature
import torch
import numpy as np
import getopt, sys

# ###############################################################    

if  __name__ == "__main__":

    proteinlist_file = "protein-kinase.list"
    with open(proteinlist_file) as proteins:
        nsamples = len(proteins.readlines())

    print(nsamples)
    y_lable = np.zeros(nsamples,dtype=int)


    esmfeature_type = "all"
   
    esmfeature_datapath = "~/PrePROTAC/featuredata/esm/data/"


    esmfeature_shape, num_esmfeature = getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath, nsamples)
    randomforest_gridsearch_ROC_train(esmfeature_shape,y_lable)
