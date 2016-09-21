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
# Delete cached data
#

if [ "$1" == "" ]
then
    echo Usage: $0 /path/to/data
    exit
fi

data_dir=$1
rm -ivrf $data_dir/train_?/tain-* $data_dir/train_?/eval-* $data_dir/train_?/full-* $data_dir/test_?/test-*
