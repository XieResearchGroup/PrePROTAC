#!/bin/bash

#for j in AAC EAAC CKSAAP DPC DDE TPC BINARY GAAC EGAAC CKSAAGP GDPC GTPC AAINDEX ZSCALE BLOSUM62 NMBroto Moran Geary CTDC CTDT CTDD CTriad KSCTriad SOCNumber QSOrder PAAC APAAC
for j in `cat features.dat`
do
	score=`grep "average" ${j}/${j}.dat | awk '{printf "%s\n",$3}'`
	echo ${j} $score >> rocscore.dat
done

