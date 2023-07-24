import csv
from gridsearchgb import gradientboost_gridsearch_ROC_train
from getdataset import getdataset_dscript
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


    esmfeature_type = "max"
   
    esmfeature_datapath = "~/PrePROTAC/featuredata/dscript/data/"


    esmfeature_shape, num_esmfeature = getdataset_dscript(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath, nsamples)
    gradientboost_gridsearch_ROC_train(esmfeature_shape,y_lable)
