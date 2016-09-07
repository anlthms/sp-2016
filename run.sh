#!/bin/bash -e
#
# Build per-subject models and combine predictions into a submission file.
#

if [ "$1" == "" ]
then
    echo Usage: $0 /path/to/data
    exit
fi

data_dir=$1
num_epochs=16
elec=0
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

    echo Processing subject $subj...
    ./model.py -e $num_epochs -w $train_dir -r 0 -z 64 -v --no_progress_bar -elec $elec
done

# Generate submission file.
./subm.py $data_dir/sample_submission.csv test.1.$elec.npy test.2.$elec.npy test.3.$elec.npy
