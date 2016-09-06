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
    def  __init__(self, data_dir, elecs=range(16)):
        self.data_dir = data_dir
        self.elecs = elecs

    def run(self, tain_path, pattern, train_percent=80):
        test_path = tain_path.replace('train', 'test')
        assert os.path.exists(tain_path)
        assert os.path.exists(test_path)
        tain_idx = os.path.join(tain_path, 'tain-index.csv')
        eval_idx = os.path.join(tain_path, 'eval-index.csv')
        test_idx = os.path.join(test_path, 'test-index.csv')
        if os.path.exists(tain_idx) and os.path.exists(eval_idx) and os.path.exists(test_idx):
            return tain_idx, eval_idx, test_idx
        self.extract(tain_path)
        self.extract(test_path)
        # Split into training and validation subsets.
        np.random.seed(0)
        with open(tain_idx, 'w') as tain_fd, open(eval_idx, 'w') as eval_fd:
            tain_fd.write('filename,label\n')
            eval_fd.write('filename,label\n')
            files = glob.glob(os.path.join(tain_path, pattern))
            files = map(os.path.basename, files)
            files = sorted(files)
            segms = np.unique([int(f.split('_')[1]) for f in files])
            np.random.shuffle(segms)
            tain_count = (len(segms) * train_percent) // 100
            tain_segms = segms[:tain_count]
            for idx, filename in enumerate(files):
                segm = int(filename.split('_')[1])
                fd = tain_fd if segm in tain_segms else eval_fd
                label = filename.split('.')[0].split('_')[-1]
                fd.write(filename + ',' + label + '\n')
        # Now create the test set.
        with open(test_idx, 'w') as test_fd:
            test_fd.write('filename,label\n')
            files = glob.glob(os.path.join(test_path, pattern))
            files = map(os.path.basename, files)
            files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[1]))
            for idx, filename in enumerate(files):
                test_fd.write(filename + ',0\n')
        return tain_idx, eval_idx, test_idx

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

        dat = mat['dataStruct'][0,0][0]
        for elec in self.elecs:
            dstfile = srcfile.replace('mat', str(elec) + '.wav')

            aud = dat[:, elec]
            min = aud.min()
            if -min > 0x8000:
                aud *= 0x8000 / -min

            max = aud.max()
            if max > 0x7FFF:
                aud *= 0x7FFF / max
            aud = np.int16(aud)
            audiolab.wavwrite(aud, dstfile, fs=400, enc='pcm16')
