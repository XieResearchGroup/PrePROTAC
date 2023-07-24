#!/bin/sh

for i in `cat target.list`
do
        echo ${i}
        grep "${i}      " human-protein.list > feature.list
        python ./print-esmfeature.py > ${i}.txt
done

