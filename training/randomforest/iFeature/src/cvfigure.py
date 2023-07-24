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

    iFeature_type = sys.argv[1]
    print(iFeature_type)

    iFeature_datapath = "~/PrePROTAC/featuredata/iFeature/data/"

    iFeature_shape, num_iFeature = getdataset_iFeature(proteinlist_file,iFeature_type,y_lable,iFeature_datapath, nsamples)

    randomforest_ROC_fpr_recallcurve(iFeature_shape,y_lable)
