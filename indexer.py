"""
Generate index files required by the neon data loader.
Also extract the data out of .mat files and save as .wav files.
"""
import os
import glob
import numpy as np
from scipy import io
from scikits import audiolab


class Indexer:
    def __init__(self, data_dir, elecs=range(16)):
        self.data_dir = data_dir
        self.elecs = elecs

    def run(self, tain_path, pattern, testing=False, train_percent=80):
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

        if testing is False:
            self.extract(tain_path)
        files = glob.glob(os.path.join(tain_path, pattern))
        files = map(os.path.basename, files)
        files = sorted(files)

        if testing is True:
            self.extract(test_path)
            with open(tain_idx, 'w') as tain_fd:
                tain_fd.write('filename,label\n')
                for filename in files:
                    label = tokenize(filename)[-1]
                    tain_fd.write(filename + ',' + label + '\n')
            with open(test_idx, 'w') as test_fd:
                test_fd.write('filename,label\n')
                files = glob.glob(os.path.join(test_path, pattern))
                files = map(os.path.basename, files)
                files = sorted(files, key=lambda x: int(tokenize(x)[1]))
                for idx, filename in enumerate(files):
                    test_fd.write(filename + ',0\n')
        else:
            # Split into training and validation subsets.
            np.random.seed(0)
            with open(tain_idx, 'w') as tain_fd, open(test_idx, 'w') as test_fd:
                tain_fd.write('filename,label\n')
                test_fd.write('filename,label\n')
                segms = np.unique([int(f.split('_')[1]) for f in files])
                np.random.shuffle(segms)
                tain_count = (len(segms) * train_percent) // 100
                tain_segms = segms[:tain_count]
                for filename in files:
                    segm = int(filename.split('_')[1])
                    fd = tain_fd if segm in tain_segms else test_fd
                    label = filename.split('.')[0].split('_')[-1]
                    fd.write(filename + ',' + label + '\n')
        return tain_idx, test_idx

    def extract(self, path):
        print('Extracting data into %s...' % path)
        filelist = glob.glob(os.path.join(path, '*.mat'))
        for srcfile in filelist:
            self.wavwrite(srcfile)

    def wavwrite(self, srcfile):
        try:
            mat = io.loadmat(srcfile)
        except ValueError:
            print('Could not load %s' % srcfile)
            return

        dat = mat['dataStruct'][0, 0][0]
        mn = dat.min()
        mx = dat.max()
        mx = max(abs(mx), abs(mn))
        if mx != 0:
            dat *= 0x7FFF / mx
        dat = np.int16(dat)

        for elec in self.elecs:
            dstfile = srcfile.replace('mat', str(elec) + '.wav')
            aud = dat[:, elec]
            audiolab.wavwrite(aud, dstfile, fs=400, enc='pcm16')
