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
from prep import nwin
from util import auc, avg, avg_preds
from sklearn import metrics


def calibrate(subjid, preds):
    length = preds.shape[0]
    sorted_preds = sorted(preds)
    # These numbers are from the training set.
    if subjid == 1:
        sel = length * 1152 / (1152 + 150)
    elif subjid == 2:
        sel = length * 2196 / (2196 + 150)
    else:
        sel = length * 2244 / (2244 + 150)
    val = sorted_preds[sel]
    std = preds.std()
    preds -= val
    preds /= std


def normalize(preds):
    preds -= preds.min()
    preds /= preds.max()


if len(sys.argv) < 3:
    print('Usage: %s <data_dir> <output_dir>' % sys.argv[0])
    sys.exit(0)

data_dir = sys.argv[1]
output_dir = sys.argv[2]

# Validate
eval_preds = None
eval_labels = None
for subjid in range(1, 4):
        subj = str(subjid)
        path = os.path.join(output_dir, 'eval.' + subj + '.npy')
        preds = np.load(path)
        eval_filename = 'eval-' + subj + '-' + str(0) + '-index.csv'
        idx_file = os.path.join(data_dir, 'train_' + subj, eval_filename)
        labels = np.loadtxt(idx_file, delimiter=',', skiprows=1, usecols=[1])
        labels, preds = avg(labels, preds)
        calibrate(subjid, preds)
        print('Eval AUC for subject %d %.4f\n' % (subjid, auc(labels, preds)))
        eval_preds = preds if subjid == 1 else np.hstack((eval_preds, preds))
        eval_labels = labels if subjid == 1 else np.hstack((eval_labels, labels))
normalize(eval_preds)

print('Overall AUC %.4f\n' % auc(eval_labels, eval_preds))

# Test
preds = None
for subjid in range(1, 4):
    path = os.path.join(output_dir, 'test.' + str(subjid) + '.npy')
    vals = np.load(path)
    vals = avg_preds(vals)
    calibrate(subjid, vals)
    preds = vals if subjid == 1 else np.hstack((preds, vals))
normalize(preds)
sample_subm = os.path.join(data_dir, 'sample_submission.csv')
assert os.path.exists(sample_subm)
files = np.loadtxt(sample_subm, dtype=str, delimiter=',', skiprows=1, usecols=[0])

with open('subm.csv', 'w') as fd:
    fd.write('File,Class\n')
    for i in range(len(files)):
        fd.write(files[i])
        fd.write(',%.6e\n' % preds[i])
print('Wrote subm.csv')
