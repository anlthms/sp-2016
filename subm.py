#!/usr/bin/env python
"""
Generate submission for kaggle.
"""
import sys
import numpy as np

if len(sys.argv) < 5:
    print('Usage: %s sample_submission.csv test.1.0.npy test.2.0.npy test.3.0.npy' % sys.argv[0])
    sys.exit(0)

files = np.loadtxt(sys.argv[1], dtype=str, delimiter=',', skiprows=1, usecols=[0])
test1, test2, test3 = sys.argv[2:5]

preds = np.hstack((np.load(test1), np.load(test2), np.load(test3)))
assert preds.shape[0] == len(files)

with open('subm.csv', 'w') as fd:
    fd.write('File,Class\n')
    for i in range(len(files)):
        fd.write(files[i])
        fd.write(',')
        fd.write('%.4e' % preds[i])
        fd.write('\n')
print('Wrote subm.csv')

