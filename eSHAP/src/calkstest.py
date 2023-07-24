import csv
import torch
import numpy as np
import getopt, sys
import random
import math
import matplotlib.pyplot as plt
import os.path
from readesm import getesmFeature
import argparse
import pandas as pd

# ###############################################################    

if  __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    uniprotid = args.name

    seqfile = "~/PrePROTAC/eSHAP/" + uniprotid + "/" + uniprotid + ".fasta"

    wholeseq = ""
    seq_handle = open(seqfile)
    for seqline in seq_handle:
        if(not seqline.startswith('>')):
            wholeseq =wholeseq+ seqline.replace("\n","")

    nmutated = len(wholeseq)
    
    keyfeature = [368, 315, 370, 495, 715, 988, 900, 324, 1236, 278, 1006, 21, 590, 1179, 27, 366, 349, 187, 1027, 655]

    datapath = "~/PrePROTAC/eSHAP/data/"

    proteinfeature_file = "~/PrePROTAC/eSHAP/" + uniprotid + "/data/" + uniprotid + ".txt"

    ref_feature_list = []
    ref_feature_list, num_feature = getesmFeature(proteinfeature_file)

    esm_feature_shape = np.zeros((nmutated, num_feature))
    diff_esm_feature = np.zeros((nmutated, num_feature))


    for aa in range(nmutated):
        totmut = 0
        for j in range(11):
            proteinfeature_file = "~/PrePROTAC/eSHAP/" + uniprotid + "/data/" + uniprotid + "_" + str(aa+1)+"_"+str(j)+".txt"
            if (os.path.isfile(proteinfeature_file)):
                totmut = totmut+1
                esm_feature_list=[]
                diff_feature_list=[]
                esm_feature_list, num_feature = getesmFeature(proteinfeature_file)
                diff_feature_list = np.subtract(ref_feature_list, esm_feature_list)
                esm_feature_shape[aa]=esm_feature_list
                diff_esm_feature[aa] = np.add(diff_esm_feature[aa], diff_feature_list)
        diff_esm_feature[aa] = diff_esm_feature[aa] / totmut

    x=[]
    y=[]
    for aa in range(nmutated):
        dist = 0.0
        for f in keyfeature:
            dist = dist + diff_esm_feature[aa][f-1]*diff_esm_feature[aa][f-1]
        x.append(aa+1)
        y.append(math.sqrt(dist))

    position_importance = pd.DataFrame(list(zip(x,y)), columns=['position','position_importance'])
    position_importance.sort_values(by=['position_importance'], ascending=False,inplace=True)
    print(position_importance.to_string())
