import csv
from getdataset import getdataset_iFeature, getdataset_dscript, getdataset_esmFeature, printesmFeature
from votemodel import randomforest_esm_get_voting
import torch
import numpy as np
import getopt, sys

# ###############################################################    

if  __name__ == "__main__":

    proteinlist_file = "human-proteinpair.list"
    nsamples = 100646
    predict_class = np.zeros(nsamples)
    np.set_printoptions(threshold=np.inf)

    fitting_file = "protein-kinase.list"
    with open(fitting_file) as proteins:
        nfits = len(proteins.readlines())

    print(nfits)
    fit_lable = np.zeros(nfits,dtype=int)
    y_lable = np.zeros(nsamples,dtype=int)

    esmfeature_type = "all"
   
    esmfeature_datapath = "~/preprotac/prediction/human-protein/data/"

    fit_datapath = "~/preprotac/finalmodel/trainingkinase/data/"

    esmmodel_path = "~/preprotac/finalmodel/10-models/"

    fitting_shape, num_fitting = getdataset_esmFeature(fitting_file,esmfeature_type,fit_lable,fit_datapath, nfits)

    esmfeature_shape, num_esmfeature = getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath, nsamples)

    rf_classfier = randomforest_esm_get_voting(esmmodel_path)
    rf_classfier.fit(fitting_shape,fit_lable)
    predict_class = rf_classfier.predict_proba(esmfeature_shape)
    print(predict_class)
