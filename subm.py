#!/usr/bin/env python
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
Generate submission for kaggle.
"""
import sys
import os
import numpy as np

if len(sys.argv) < 3:
    print('Usage: %s sample_submission.csv <output_dir>' % sys.argv[0])
    sys.exit(0)

basedir = sys.argv[2]
preds = None
for subjid in range(1, 4):
    subj = str(subjid)
    for elec in range(16):
        path = os.path.join(basedir, 'test.' + subj + '.' + str(elec) + '.npy')
        vals = np.load(path)
        vals = vals.reshape((vals.shape[0], 1))
        subjpreds = vals if elec == 0 else np.hstack((subjpreds, vals))
    meanpreds = subjpreds.mean(axis=1)
    preds = meanpreds if subjid == 1 else np.hstack((preds, meanpreds))

files = np.loadtxt(sys.argv[1], dtype=str, delimiter=',', skiprows=1, usecols=[0])
assert preds.shape[0] == len(files)

with open('subm.csv', 'w') as fd:
    fd.write('File,Class\n')
    for i in range(len(files)):
        fd.write(files[i])
        fd.write(',')
        fd.write('%.4e' % preds[i])
        fd.write('\n')
print('Wrote subm.csv')
