import csv
import torch
import numpy as np
import getopt, sys
import os.path
import argparse

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


    torch.set_printoptions(profile="full")
    for i in range(len(wholeseq)):
        for j in range(11):
            proteinfeature_file = "~/PrePROTAC/eSHAP/" + uniprotid + "/pt_data/" + uniprotid + "_" + str(i+1)+"_"+str(j)+".pt"
            if (os.path.isfile(proteinfeature_file)):
                proteinfeature_outfile = "~/PrePROTAC/eSHAP/" + uniprotid + "/data/" + uniprotid + "_" + str(i+1)+"_"+str(j)+".txt"
                feature_data = torch.load(proteinfeature_file)
                f = open(proteinfeature_outfile, 'w')
                print(feature_data, file=f)
                f.close()
