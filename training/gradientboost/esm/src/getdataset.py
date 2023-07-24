import csv
import torch
import numpy as np
import getopt, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold

#from sklearn.experimental import enable_halving_search_cv  # noqa
#from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd

def getdataset_iFeature(namelist_file, feature_type, y_lable, datapath):


    nsamples = 444
    namelist_handle = open(namelist_file)
    nprotein = 0
    num_feature = 0
    for namelist_line in namelist_handle:
        protein_name = namelist_line.split("\t")[0]
        protein_label = namelist_line.split("\t")[1]
        proteinfeature_file = datapath + protein_name + "-" + feature_type + ".dat"
        y_lable[nprotein]=int(protein_label)
        AAC_feature_list=[]

        AAC_feature_handle = open(proteinfeature_file)
        for feature_line in AAC_feature_handle:
            #print(feature_line)
            if feature_line.startswith(protein_name):
                tmp = feature_line.split("\t")
                tmp_feature = 0
                for xtmp in tmp[1:]:
                    tmp_feature = tmp_feature + 1
                    ifeature = float(xtmp)
                    AAC_feature_list.append(ifeature)

        AAC_feature_handle.close()

        if(nprotein == 0):
                num_feature = tmp_feature
                AAC_feature_shape = np.zeros((nsamples, num_feature))

        AAC_feature_shape[nprotein]=AAC_feature_list

        nprotein = nprotein + 1

        if(num_feature!=tmp_feature):
            print("the feature for ", nprotein, " is not correct! ", num_feature, tmp_feature)


    return AAC_feature_shape, num_feature


def getdataset_dscript(namelist_file, feature_type, y_lable, feature_datapath):


    nsamples = 444
    namelist_handle = open(namelist_file)
    nprotein = 0
    num_feature = 0
    for namelist_line in namelist_handle:
        protein_name = namelist_line.split("\t")[0]
        protein_label = namelist_line.split("\t")[1]
        proteinfeature_max_file = feature_datapath + protein_name + ".max.dat"
        proteinfeature_avg_file = feature_datapath + protein_name + ".avg.dat"
        y_lable[nprotein]=int(protein_label)
        max_feature_list=[]
        avg_feature_list=[]

        if(feature_type == "max" or feature_type == "all"):

            max_feature_handle = open(proteinfeature_max_file)
            n_feature = 0
            for feature_line in max_feature_handle:
                if feature_line.startswith("tensor([[["):
                    n_feature = n_feature + 1
                    tmp = feature_line.split("[")
                    max_feature = float(tmp[4].split("]")[0])
                    max_feature_list.append(max_feature)
                    continue
                if n_feature > 0:
                    n_feature = n_feature + 1
                    tmp = feature_line.split("[")
                    max_feature = float(tmp[1].split("]")[0])
                    max_feature_list.append(max_feature)
            max_feature_handle.close()

            if(nprotein == 0):
                num_feature = n_feature
                max_feature_shape = np.zeros((nsamples, num_feature))

            max_feature_shape[nprotein]=max_feature_list

        if(feature_type == "avg" or feature_type == "all"):
            avg_feature_handle = open(proteinfeature_avg_file)
            n_feature = 0
            for feature_line in avg_feature_handle:
                if feature_line.startswith("tensor([[["):
                    n_feature = n_feature + 1
                    tmp = feature_line.split("[")
                    avg_feature = float(tmp[4].split("]")[0])
                    avg_feature_list.append(avg_feature)
                    continue
                if n_feature > 0:
                    n_feature = n_feature + 1
                    tmp = feature_line.split("[")
                    avg_feature = float(tmp[1].split("]")[0])
                    avg_feature_list.append(avg_feature)
            avg_feature_handle.close()

            if(nprotein == 0):
                num_feature = n_feature
                avg_feature_shape = np.zeros((nsamples, num_feature))

            avg_feature_shape[nprotein]=avg_feature_list

        nprotein = nprotein + 1

        if(num_feature!=n_feature):
            print("the feature for ", nprotein, " is not correct! ", num_feature, n_feature)


    if(feature_type == "max"):
        return max_feature_shape, num_feature  

    if(feature_type == "avg"):
        return avg_feature_shape, num_feature  

    if(feature_type == "all"):
        all_feature_shape = np.zeros((nsamples, 2*num_feature))
        all_feature_shape = torch.cat((max_feature_shape, avg_feature_shape), 1)
        return all_feature_shape, 2*num_feature  

    #print(max_feature_shape)
    #print(y_lable)

# ###############################################################    
def printesmFeature(namelist_file, feature_type, y_lable, datapath):


    nsamples = 444
    namelist_handle = open(namelist_file)
    nprotein = 0
    num_feature = 0
    torch.set_printoptions(profile="full")
    for namelist_line in namelist_handle:
        protein_name = namelist_line.split("\t")[0]
        protein_label = namelist_line.split("\t")[1]
        y_lable[nprotein]=int(protein_label)
        proteinfeature_file = datapath + protein_name + ".pt"
        feature_data = torch.load(proteinfeature_file)
        print(feature_data)
        nprotein = nprotein + 1

# ###############################################################    
def getdataset_esmFeature(namelist_file, feature_type, y_lable, datapath, nproteins):


    #nsamples = 444
    nsamples = nproteins
    namelist_handle = open(namelist_file)
    nprotein = 0
    num_feature = 0
    torch.set_printoptions(profile="full")
    for namelist_line in namelist_handle:
        protein_name = namelist_line.split("\t")[0]
        protein_label = namelist_line.split("\t")[1]
        y_lable[nprotein]=int(protein_label)
        proteinfeature_file = datapath + protein_name + ".txt"
        esm_feature_list=[]
        esm_feature_handle = open(proteinfeature_file)
        for feature_line in esm_feature_handle: 
            if feature_line.startswith("{"):
                tmp = feature_line.split("[")
                tmp_feature = 0
                tmp1 = tmp[1]
                tmp2 = tmp1.split(",")

            else:
                if "]" in feature_line:
                    tmp = feature_line.split("]")
                    tmp1 = tmp[0]
                    tmp2 = tmp1.split(",")
                else:    
                    tmp2 = feature_line.split(",")

            for xtmp in tmp2[0:5]:
                #print(xtmp)
                tmp_feature = tmp_feature + 1
                esmfeature = float(xtmp)
                esm_feature_list.append(esmfeature)
                #print(esmfeature)

        esm_feature_handle.close()

        if(nprotein == 0):
            num_feature = tmp_feature
            esm_feature_shape = np.zeros((nsamples, num_feature))

        esm_feature_shape[nprotein]=esm_feature_list

        nprotein = nprotein + 1

        if(num_feature!=tmp_feature):
            print("the feature for ", nprotein, " is not correct! ", num_feature, tmp_feature)

    print(nprotein,num_feature) 
    #print(esm_feature_shape)

    return esm_feature_shape, num_feature

# ###############################################################    
if  __name__ == "__main__":

    print("get dataset\n")
