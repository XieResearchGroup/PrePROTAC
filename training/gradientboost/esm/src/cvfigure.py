import csv
from getdataset import getdataset_iFeature, getdataset_dscript, getdataset_esmFeature, printesmFeature
from showfigure import gradientboost_ROC_fpr_recallcurve, randomforest_ROC_fpr_recallcurve, ridge_ROC_fpr_recallcurve
import torch
import numpy as np
import getopt, sys

# ###############################################################    

if  __name__ == "__main__":

    proteinlist_file = "protein-kinase.list"
    with open(proteinlist_file) as proteins:
        nsamples = len(proteins.readlines())

    print(nsamples)
    nfeatures = 20 
    num_feature = 0
    y_lable = np.zeros(nsamples,dtype=int)

    esmfeature_type = "all"

    esmfeature_datapath = "/home/lixie/work/protac/training/CRBN/newversion/esm/444-traingset/featuredata/esm/data/"

    esmfeature_shape, num_esmfeature = getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath, nsamples)

    gradientboost_ROC_fpr_recallcurve(esmfeature_shape,y_lable)
