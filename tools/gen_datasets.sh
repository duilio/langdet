#!/bin/bash

lines=$1
((half = lines/2))
awk '{print rand() "\t" $0}' | sort -S1G -nk1 | head -n $lines | cut -f 2- | split -a 1 -l $half - tmp.
cat tmp.a >> train.txt
cat tmp.b >> test.txt
rm tmp.*
