#!/usr/bin/env python
"""
Extract the data out of .mat files and save as .wav files.
"""

import sys
import os
import glob
import numpy as np
from scipy import io
from scikits import audiolab


def extract(path):
    print('Extracting data into %s...' % path)
    files = glob.glob(os.path.join(path, '*.mat'))
    assert len(files) > 0, 'No .mat files found in %s' % path
    for srcfile in files:
        wavwrite(srcfile)


def wavwrite(srcfile):
    try:
        mat = io.loadmat(srcfile)
    except ValueError:
        print('Could not load %s' % srcfile)
        return

    dat = mat['dataStruct'][0, 0][0]
    mn = dat.min()
    mx = dat.max()
    mx = float(max(abs(mx), abs(mn)))
    if mx != 0:
        dat *= 0x7FFF / mx
    dat = np.int16(dat)

    for elec in range(16):
        dstfile = srcfile.replace('mat', str(elec) + '.wav')
        aud = dat[:, elec]
        audiolab.wavwrite(aud, dstfile, fs=400, enc='pcm16')


if len(sys.argv) < 2:
    print('Usage %s /path/to/data' % sys.argv[0])
    sys.exit(0)

for subj_id in range(1, 4):
    for prefix in ['train_', 'test_']:
        path = os.path.join(sys.argv[1], prefix + str(subj_id))
        extract(path)
