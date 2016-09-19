#!/usr/bin/env python
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
