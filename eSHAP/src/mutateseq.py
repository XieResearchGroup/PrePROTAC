import csv
import torch
import numpy as np
import getopt, sys
import random
import argparse

# ###############################################################    

if  __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    uniprotid = args.name

    seqfile =  uniprotid + ".fasta"

    wholeseq = ""

    polaraa = ['S', 'T', 'N', 'Q', 'C', 'P', 'R', 'H', 'K', 'D', 'E']
    hydrophobicaa = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G']

    seq_handle = open(seqfile)
    for seqline in seq_handle:
        if(not seqline.startswith('>')):
            wholeseq =wholeseq+ seqline.replace("\n","")


    print(wholeseq)

    for aa in range(len(wholeseq)):
        #print(wholeseq[aa])
        mutated = 'A'
        if(wholeseq[aa] in polaraa):
            for mm in range(len(hydrophobicaa)): 
                mutated = hydrophobicaa[mm]
                mutatedseq = wholeseq[:aa] + mutated + wholeseq[aa+1:]
                mutatedfile = uniprotid+"_"+str(aa+1)+"_"+str(mm)+".fasta"
                f = open(mutatedfile, 'w')
                print(">"+uniprotid+"_"+str(aa+1)+"_"+str(mm), file=f)
                print(mutatedseq, file=f)
                f.close()

        if(wholeseq[aa] in hydrophobicaa):
            for mm in range(len(polaraa)): 
                mutated = polaraa[mm]
                mutatedseq = wholeseq[:aa] + mutated + wholeseq[aa+1:]
                mutatedfile = uniprotid+"_"+str(aa+1)+"_"+str(mm)+".fasta"
                f = open(mutatedfile, 'w')
                print(">"+uniprotid+"_"+str(aa+1)+"_"+str(mm), file=f)
                print(mutatedseq, file=f)
                f.close()
