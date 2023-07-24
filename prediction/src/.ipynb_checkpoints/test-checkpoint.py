import csv
from gridsearchrf import randomforest_nestedCV_train, randomforest_gridsearch_ROC_train
from getdataset import getdataset_iFeature, getdataset_dscript, getdataset_esmFeature, printesmFeature
from rocsearch import randomforest_ROCscore, randomforest_ROCcurve, randomforest_TROCcurve, randomforest_ReROCcurve
from shapsearch import rf_getshapvalue 
import torch
import numpy as np
import getopt, sys

# ###############################################################    

if  __name__ == "__main__":

    proteinlist_file = "protein-kinase.list"
    nsamples = 444
    nfeatures = 20 
    num_feature = 0
    y_lable = np.zeros(nsamples,dtype=int)

    #ifeature_type = sys.argv[1]
    #dscript_feature_type = sys.argv[2]

    dscript_feature_type = "max"
    GTPCfeature_type = "GTPC"
    Gearyfeature_type = "Geary"
    #Gearyfeature_type = "Moran"
    CTriadfeature_type = "CTriad"
    esmfeature_type = "all"
   
    ifeature_datapath = "/workspace/rftrain/data/kinase_domain/training/CRBN/iFeature/data/"
    dscript_feature_datapath = "/workspace/rftrain/data/kinase_domain/training/CRBN/D-script-feature/contactfeature/"
    esmfeature_datapath = "/workspace/rftrain/data/kinase_domain/training/CRBN/esm/data/"
    printesm_datapath = "/workspace/rftrain/data/kinase_domain/training/CRBN/esm/pt_data/"


    #GTPCfeature_shape, num_GTPCfeature = getdataset_iFeature(proteinlist_file,GTPCfeature_type,y_lable,ifeature_datapath)
    esmfeature_shape, num_esmfeature = getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath)
    #getdataset_esmFeature(proteinlist_file,esmfeature_type,y_lable,esmfeature_datapath)
    #printesmFeature(proteinlist_file,esmfeature_type,y_lable,printesm_datapath)
    #Gearyfeature_shape, num_Gearyfeature = getdataset_iFeature(proteinlist_file,Gearyfeature_type,y_lable,ifeature_datapath)
    CTriadfeature_shape, num_CTriadfeature = getdataset_iFeature(proteinlist_file,CTriadfeature_type,y_lable,ifeature_datapath)

    dscript_feature_shape, num_dsfeature = getdataset_dscript(proteinlist_file,dscript_feature_type,y_lable,dscript_feature_datapath)

    #GTPC_Gearyfeature_shape = np.zeros((nsamples,num_GTPCfeature+num_Gearyfeature))
    #GTPC_Gearyfeature_shape = np.concatenate((GTPCfeature_shape,Gearyfeature_shape),axis=1)

    #num_ifeature = num_GTPCfeature+num_Gearyfeature+num_CTriadfeature
    #ifeature_shape = np.zeros((num_ifeature))
    #ifeature_shape = np.concatenate((GTPC_Gearyfeature_shape,CTriadfeature_shape),axis=1)
    
    #print(num_ifeature, num_dsfeature)
    #comfeature_shape = np.zeros((nsamples, num_esmfeature+num_dsfeature))
    #comfeature_shape=np.concatenate((esmfeature_shape,dscript_feature_shape),axis=1)
    comfeature_shape = np.zeros((nsamples, num_esmfeature+num_CTriadfeature))
    comfeature_shape=np.concatenate((esmfeature_shape,CTriadfeature_shape),axis=1)
    #randomforest_ReROCcurve(esmfeature_shape,y_lable)
    #randomforest_ReROCcurve(comfeature_shape,y_lable)
    rf_getshapvalue(comfeature_shape,y_lable)
    #randomforest_gridsearch_ROC_train(esmfeature_shape,y_lable)

