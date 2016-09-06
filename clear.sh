#!/bin/bash
#
# Delete cached data
#

if [ "$1" == "" ]
then
    echo Usage: $0 /path/to/data
    exit
fi

data_dir=$1
rm -ivr $data_dir/train_?/tain-* $data_dir/train_?/eval-* $data_dir/train_?/full-* $data_dir/test_?/test-*
