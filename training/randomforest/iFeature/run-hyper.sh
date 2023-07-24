#!/bin/sh

for j in AAC EAAC CKSAAP DPC DDE TPC BINARY GAAC EGAAC CKSAAGP GDPC AAINDEX ZSCALE BLOSUM62 NMBroto Moran CTDC CTDT CTDD KSCTriad SOCNumber QSOrder PAAC APAAC
do
	echo ${j}
	cd ${j}
	python ./src/hypertrainesm.py ${j} | tee tune.dat
	cd ../
done

