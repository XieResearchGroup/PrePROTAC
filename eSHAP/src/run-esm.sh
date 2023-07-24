#!/bin/sh 


CUDA_VISIBLE_DEVICES=1 python  ~/esm/extract.py esm1_t34_670M_UR50S BRD4_mutated.fasta ./pt_data --repr_layers 34 --include mean
