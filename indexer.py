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
    def __init__(self, repo_dir, subj, validate_mode, training):
        self.repo_dir = repo_dir
        self.subj = subj
        self.validate_mode = validate_mode
        self.training = training

    def get_filename(self, tain_path, test_path, elec):
        def get_path(basepath, name, elec):
            filename = name + '-' + str(self.subj) + '-' + str(elec) + '-index.csv'
            return os.path.join(basepath, filename)

        assert os.path.exists(tain_path), 'Path not found: %s' % tain_path
        if self.training:
            if self.validate_mode:
                idx_file = get_path(tain_path, 'tain', elec)
            else:
                idx_file = get_path(tain_path, 'full', elec)
        else:
            if self.validate_mode:
                idx_file = get_path(tain_path, 'eval', elec)
            else:
                assert os.path.exists(test_path), 'Path not found: %s' % test_path
                idx_file = get_path(test_path, 'test', elec)
        return idx_file

    def run(self, elec, train_percent=70):
        def tokenize(filename):
            return filename.split('.')[0].split('_')

        def get_segm(filename):
            return int(tokenize(filename)[1])

        def get_label(filename):
            return tokenize(filename)[-1]

        tain_path = self.repo_dir
        test_path = tain_path.replace('train', 'test')

        idx_file = self.get_filename(tain_path, test_path, elec)
        if os.path.exists(idx_file):
            return idx_file

        print('Creating %s...' % idx_file)
        path = tain_path if (self.training or self.validate_mode) else test_path
        pattern = '*.' + str(elec) + '.wav'
        files = glob.glob(os.path.join(path, pattern))
        assert len(files) > 0, 'No .wav files found in %s' % path
        files = sorted(map(os.path.basename, files))
        files = sorted(files, key=lambda x: get_segm(x))

        if self.training:
            np.random.seed(0)
            np.random.shuffle(files)

        labels = []
        if self.validate_mode:
            # Split into training and validation subsets.
            segms = np.unique([get_segm(f) for f in files])
            # Make sure that segments from the same hour go into the same subset.
            hours = range(segms.min(), segms.max() + 1, 6)
            np.random.seed(0)
            np.random.shuffle(hours)
            tain_count = (len(hours) * train_percent) // 100
            chosen_hours = hours[:tain_count] if self.training else hours[tain_count:]
            chosen_files = []
            for filename in files:
                segm = get_segm(filename)
                hour = segm - (segm - 1) % 6
                if hour in chosen_hours:
                    chosen_files.append(filename)
                    labels.append(get_label(filename))
        else:
            chosen_files = files
            for filename in files:
                label = get_label(filename) if self.training else '0'
                labels.append(label)

        with open(idx_file, 'w') as fd:
            fd.write('filename,label\n')
            for filename, label in zip(chosen_files, labels):
                fd.write(filename + ',' + label + '\n')
        return idx_file
