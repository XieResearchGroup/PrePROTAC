import csv
import torch
import numpy as np
import getopt, sys
import pandas as pd

# ###############################################################    
def getesmFeature(proteinfeature_file):


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

    num_feature = tmp_feature
    esm_feature_handle.close()

    return esm_feature_list, num_feature

# ###############################################################    
if  __name__ == "__main__":

    print("get dataset\n")
