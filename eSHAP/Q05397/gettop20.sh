#!/bin/sh

head -21 Q05397.position | tail -20 > 1.dat
cat -n 1.dat > 2.dat
cat 2.dat | awk '{printf "%10s %10s %10d %10s\n", $1, $3, $3+421, $4}' > 3.dat
mv 3.dat top20.position
cat top20.position | awk '{printf "%s %d %s ", "resi", $3, "or"}' > Q05397.pml
rm -f 1.dat 2.dat
