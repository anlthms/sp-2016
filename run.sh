#!/bin/bash
#
# Build per-subject models for each electrode and combine predictions into a submission file.
#

if [ "$2" == "" ]
then
    echo Usage: $0 /path/to/data /path/to/output
    exit
fi

data_dir=$1
out_dir=$2
num_epochs=16
set -x

if [ -f $data_dir/prepdone ]
then
    echo $data_dir/prepdone exists. Skipping prep...
else
    ./prep.py $data_dir
    touch $data_dir/prepdone
fi

for elec in `seq 0 15`
do
    echo Electrode $elec...
    ./clear.sh $data_dir
    for subj in `seq 1 3`
    do
        train_dir=$data_dir/train_$subj/

        if [ ! -e $train_dir ]
        then
            echo $train_dir not found!
            exit
        fi

        echo Processing subject $subj...
        # Validate
        ./model.py -e $num_epochs -w $train_dir -r 0 -z 64 -v --no_progress_bar -elec $elec -out $out_dir
        # Test
        ./model.py -e $num_epochs -w $train_dir -r 0 -z 64 -v --no_progress_bar -elec $elec -out $out_dir -test
    done
done

./subm.py sample_submission.csv $out_dir
