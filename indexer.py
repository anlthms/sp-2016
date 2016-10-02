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
"""
Generate index files required by the neon data loader.
"""
import os
import glob
import numpy as np


class Indexer:
    def __init__(self):
        pass

    def run(self, tain_path, pattern, testing=False, train_percent=70):
        def tokenize(filename):
            return filename.split('.')[0].split('_')

        assert os.path.exists(tain_path)
        if testing is True:
            test_path = tain_path.replace('train', 'test')
            assert os.path.exists(test_path)
            tain_idx = os.path.join(tain_path, 'full-index.csv')
            test_idx = os.path.join(test_path, 'test-index.csv')
        else:
            tain_idx = os.path.join(tain_path, 'tain-index.csv')
            test_idx = os.path.join(tain_path, 'eval-index.csv')

        if os.path.exists(tain_idx) and os.path.exists(test_idx):
            return tain_idx, test_idx

        files = glob.glob(os.path.join(tain_path, pattern))
        assert len(files) > 0, 'No .wav files found in %s' % tain_path
        files = map(os.path.basename, files)
        files = sorted(files)

        np.random.seed(0)
        np.random.shuffle(files)
        if testing is True:
            with open(tain_idx, 'w') as tain_fd:
                tain_fd.write('filename,label\n')
                for filename in files:
                    label = tokenize(filename)[-1]
                    tain_fd.write(filename + ',' + label + '\n')
            with open(test_idx, 'w') as test_fd:
                test_fd.write('filename,label\n')
                files = glob.glob(os.path.join(test_path, pattern))
                assert len(files) > 0, 'No .wav files found in %s' % test_path
                files = map(os.path.basename, files)
                files = sorted(files, key=lambda x: int(tokenize(x)[1]))
                for filename in files:
                    test_fd.write(filename + ',0\n')
        else:
            # Split into training and validation subsets.
            with open(tain_idx, 'w') as tain_fd, open(test_idx, 'w') as test_fd:
                tain_fd.write('filename,label\n')
                test_fd.write('filename,label\n')
                segms = np.unique([int(f.split('_')[1]) for f in files])
                hours = range(segms.min(), segms.max() + 1, 6)
                np.random.shuffle(hours)
                tain_count = (len(hours) * train_percent) // 100
                tain_hours = hours[:tain_count]
                for filename in files:
                    segm = int(filename.split('_')[1])
                    hour = segm - (segm - 1) % 6
                    fd = tain_fd if hour in tain_hours else test_fd
                    label = tokenize(filename)[-1]
                    fd.write(filename + ',' + label + '\n')
        return tain_idx, test_idx
