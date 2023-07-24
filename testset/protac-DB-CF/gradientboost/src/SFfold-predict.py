import csv
from getdataset import getdataset_iFeature, getdataset_dscript, getdataset_esmFeature, printesmFeature
from predictrocfpr import predict_ROC_fpr_recall
from sklearn.model_selection import RepeatedStratifiedKFold
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
   
    esmfeature_datapath = "~/preprotac/testset/protac-DB-CF/data/"

    esmmodel_path = "~/preprotac/testset/model/"
    esmmodel_file = "gradientboost-esm-proteinkinase.pkl"


    esmfeature_shape, num_esmfeature = getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath, nsamples)	

    for i in range(10):

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

        for j, (train, test) in enumerate(cv.split(esmfeature_shape, y_lable)):

            predict_ROC_fpr_recall(esmfeature_shape[train], y_lable[train], esmmodel_path, esmmodel_file)
