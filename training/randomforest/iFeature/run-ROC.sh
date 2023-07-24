#!/bin/sh

#for j in AAC EAAC CKSAAP DPC DDE TPC BINARY GAAC EGAAC CKSAAGP GDPC GTPC AAINDEX ZSCALE BLOSUM62 NMBroto Moran Geary CTDC CTDT CTDD CTriad KSCTriad SOCNumber QSOrder PAAC APAAC
for j in `cat features.dat`
do
	#mkdir ${j}
	echo ${j}
	cd ${j}
	#cp ../protein-kinase.list ./
	python ./src/cvfigure.py ${j} > ${j}.dat
	cd ../
done

