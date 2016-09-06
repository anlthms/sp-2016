#!/bin/bash

if [ "$1" == "" ]
then
    echo Usage:  $0 /path/to/data
    exit
fi

data_dir=$1
num_epochs=16
set -x

# Train and predict for each subject.
for subj in `seq 1 3`
do
    train_dir=$data_dir/train_$subj/

    if [ ! -e $train_dir ]
    then
        echo $train_dir not found!
        exit
    fi

    ./model.py -e $num_epochs -w $train_dir -r 0 -eval 1 -v -elec 0 -z64
done

# Generate submission file.
./subm.py $data_dir/sample_submission.csv test.1.0.npy test.2.0.npy test.3.0.npy  
