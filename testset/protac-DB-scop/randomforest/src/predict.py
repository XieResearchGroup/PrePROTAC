import csv
from getdataset import getdataset_iFeature, getdataset_dscript, getdataset_esmFeature, printesmFeature
from predictrocfpr import predict_ROC_fpr_recall
import numpy as np
import getopt, sys

# ###############################################################    

if  __name__ == "__main__":

    proteinlist_file = "predict.list"
    with open(proteinlist_file) as proteins:
        nsamples = len(proteins.readlines())

    y_lable = np.zeros(nsamples,dtype=int)
    E3_lable = np.zeros(nsamples,dtype=str)
    predict_class = np.zeros(nsamples)
    np.set_printoptions(threshold=np.inf)


    esmfeature_type = "all"
   
    esmfeature_datapath = "~/preprotac/testset/protac-DB-scop/data/"

    esmmodel_path = "~/preprotac/testset/model/"
    esmmodel_file = "random-forest-esm-proteinkinase.pkl"


    esmfeature_shape, num_esmfeature = getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath, nsamples)	

    predict_ROC_fpr_recall(esmfeature_shape, y_lable, esmmodel_path, esmmodel_file)
