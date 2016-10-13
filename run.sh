#!/bin/bash -e
#
#   Copyright 2016 Anil Thomas
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

#
# Build per-subject models and then combine predictions into a submission file.
#

if [ "$2" == "" ]
then
    echo Usage: $0 /path/to/data /path/to/output
    exit
fi

data_dir=$1
out_dir=$2
num_epochs=4
bsz=64
prep_file=$data_dir/prepdone
set -x

if [ -f $prep_file ]
then
    echo $prep_file exists. Skipping prep...
else
    ./clear.sh $data_dir
    ./prep.py $data_dir
    touch $prep_file
fi

for elec in `seq 0 15`
do
    echo Electrode $elec...
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
        ./model.py -e $num_epochs -w $train_dir -r 1 -z $bsz -v --no_progress_bar -elec $elec -out $out_dir -eval 1 -validate
        # Test
        ./model.py -e $num_epochs -w $train_dir -r 1 -z $bsz -v --no_progress_bar -elec $elec -out $out_dir
    done
done

./subm.py sample_submission.csv $out_dir
