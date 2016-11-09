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
        if training or validate_mode:
            self.max_rep_count = 3
            self.safety_flags = {}
            par_dir = os.path.dirname(os.path.normpath(repo_dir))
            safety_file = os.path.join(par_dir, 'train_and_test_data_labels_safe.csv')
            safety_info = np.loadtxt(safety_file, dtype=str, delimiter=',', skiprows=1)
            self.safety_flags = {line[0].split('.')[0]: line[2] == '1' for line in safety_info}

    def make_filename(self, path, set_name):
        assert os.path.exists(path), 'Path not found: %s' % path
        filename = set_name + '-index.csv'
        idx_file = os.path.join(path, filename)
        return idx_file

    def tokenize(self, filename):
        return filename.split('.')[0].split('_')

    def get_segm(self, filename):
        return int(self.tokenize(filename)[1])

    def get_label(self, filename):
        return int(self.tokenize(filename)[-1])

    def is_safe(self, filename):
        assert self.training is True or self.validate_mode is True
        key = os.path.basename(filename).split('.')[0]
        if key in self.safety_flags:
            return self.safety_flags[key]
        return False

    def run(self, elec, set_name):
        tain_path = self.repo_dir
        test_path = tain_path.replace('train', 'test') + '_new'
        path = tain_path if (self.training or self.validate_mode) else test_path
        idx_file = self.make_filename(path, set_name)
        if os.path.exists(idx_file):
            return idx_file

        print('Creating %s...' % idx_file)
        pattern = '*.' + str(elec) + '.wav'
        files = glob.glob(os.path.join(path, pattern))
        assert len(files) > 0, 'No .wav files found in %s' % path
        if self.training or self.validate_mode:
            files = filter(lambda x: self.is_safe(x), files)
        files = sorted(map(os.path.basename, files))
        files = sorted(files, key=lambda x: self.get_segm(x))

        if self.training or self.validate_mode:
            chosen_files, labels = self.choose(files)
        else:
            chosen_files, labels = files, np.zeros(len(files))

        if self.training and not self.validate_mode:
            self.append_old_test(chosen_files, labels, elec)

        with open(idx_file, 'w') as fd:
            fd.write('filename,label\n')
            for filename, label in zip(chosen_files, labels):
                fd.write(filename + ',' + str(label) + '\n')
        return idx_file

    def append_old_test(self, files, labels, elec):
        path = self.repo_dir.replace('train', 'test')
        pattern = '*.' + str(elec) + '.wav'
        test_files = glob.glob(os.path.join(path, pattern))
        test_files = sorted(map(os.path.basename, test_files))
        for filename in test_files:
            if not self.is_safe(filename):
                continue
            filename = os.path.join(os.path.pardir, os.path.basename(path),
                                    os.path.basename(filename))
            for i in range(self.max_rep_count + 2):
                files.append(filename)
                labels.append(1)

    def choose(self, files):
        train_percent = 70 if self.validate_mode else 100
        nfiles = []
        pfiles = []
        for fname in files:
            if self.get_label(fname) == 0:
                nfiles.append(fname)
            else:
                pfiles.append(fname)
        psegms = np.unique([self.get_segm(f) for f in pfiles])
        nsegms = np.unique([self.get_segm(f) for f in nfiles])
        pmax = (np.max(psegms) * train_percent) // 100
        nmax = (np.max(nsegms) * train_percent) // 100

        chosen_files = []
        labels = []
        for filename in files:
            label = self.get_label(filename)
            max_hour = nmax if label == 0 else pmax
            segm = self.get_segm(filename)
            hour = segm - (segm - 1) % 6
            if self.training and hour > max_hour:
                continue
            if not self.training and hour <= max_hour:
                continue
            if self.training:
                # Repeat recent samples.
                rep_count = ((self.max_rep_count * hour) / max_hour) + 1
            else:
                rep_count = 1
            for i in range(rep_count):
                chosen_files.append(filename)
                labels.append(label)
        return chosen_files, labels
