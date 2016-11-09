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
Extract the data out of .mat files and save as .wav files.
"""

import sys
import os
import glob
import numpy as np
from scipy import io, signal
from scikits import audiolab


# Downsampling factor
ds_factor = 1
# Window duration in minutes
win_dur = 10
# Number of windows (assuming a stride of 1 minute)
nwin = 10 - win_dur + 1


def extract(path, fs, training):
    print('Extracting data into %s...' % path)
    files = glob.glob(os.path.join(path, '*.mat'))
    assert len(files) > 0, 'No .mat files found in %s' % path
    for srcfile in files:
        wavwrite(srcfile, fs, training)


def wavwrite(srcfile, fs, training):
    try:
        mat = io.loadmat(srcfile)
    except ValueError:
        print('Could not load %s' % srcfile)
        return

    dat = mat['dataStruct'][0, 0][0]
    if ds_factor != 1:
        dat = signal.decimate(dat, ds_factor, axis=0, zero_phase=True)

    mn = dat.min()
    mx = dat.max()
    mx = float(max(abs(mx), abs(mn)))
    if training and mx == 0:
        print('skipping %s' % srcfile)
        return
    if mx != 0:
        dat *= 0x7FFF / mx
    dat = np.int16(dat)

    winsize = win_dur * 60 * fs
    stride = 60 * fs
    for elec in range(16):
        aud = dat[:, elec]
        for win in range(nwin):
            dstfile = srcfile.replace('mat', str(win) + '.' + str(elec) + '.wav')
            beg = win * stride
            end = beg + winsize
            clip = aud[beg:end]
            audiolab.wavwrite(clip, dstfile, fs=fs, enc='pcm16')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage %s /path/to/data' % sys.argv[0])
        sys.exit(0)

    fs = 400
    if ds_factor != 1:
        fs /= ds_factor
    for subj_id in range(1, 4):
        for template in ['train_%d', 'test_%d', 'test_%d_new']:
            path = os.path.join(sys.argv[1], template % subj_id)
            extract(path, fs, template == 'train_%d')
