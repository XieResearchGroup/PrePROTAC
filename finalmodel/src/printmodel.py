import csv
from getdataset import getdataset_iFeature, getdataset_dscript, getdataset_esmFeature, printesmFeature
from showfigure import randomforest_ROC_fpr_recallcurve
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
    E3_lable = np.zeros(nsamples,dtype=str)

    esmfeature_type = "all"
    preprotac_path = "~/preprotac/"
   
    esmfeature_datapath = preprotac_path + "finalmodel/trainingkinase/data/"

    esmfeature_shape, num_esmfeature = getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath, nsamples)

    randomforest_ROC_fpr_recallcurve(esmfeature_shape,y_lable)
