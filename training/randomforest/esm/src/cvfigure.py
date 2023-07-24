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

    y_lable = np.zeros(nsamples,dtype=int)

    esmfeature_type = "all"

    esmfeature_datapath = "~/PrePROTAC/featuredata/esm/data/"

    esmfeature_shape, num_esmfeature = getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath, nsamples)

    randomforest_ROC_fpr_recallcurve(esmfeature_shape,y_lable)
