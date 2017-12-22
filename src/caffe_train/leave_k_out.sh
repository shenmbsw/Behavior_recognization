#!/bin/bash
leaveout="1"

cp -r all/* train/ 

mv train/0/*_"$leaveout"_*_* test/0/
mv train/1/*_"$leaveout"_*_* test/1/
mv train/2/*_"$leaveout"_*_* test/2/
mv train/3/*_"$leaveout"_*_* test/3/
mv train/4/*_"$leaveout"_*_* test/4/
mv train/5/*_"$leaveout"_*_* test/5/
mv train/6/*_"$leaveout"_*_* test/6/

python data_aug.py
python make_train.py
