#!/usr/bin/env python
"""
Train a per-subject model for:
https://www.kaggle.com/c/melbourne-university-seizure-prediction

Usage:
    ./model.py -e 16 -w </path/to/data> -r 0 -z 64 -elec <electrode index>
"""

import os
import glob
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian, GlorotUniform
from neon.layers import Conv, Pooling, GeneralizedCost, Affine
from neon.layers import DeepBiRNN, Reshape
from neon.optimizers import Adagrad
from neon.transforms import Rectlin, Softmax, CrossEntropyBinary
from neon.models import Model
from neon.data import DataLoader, AudioParams
from neon.callbacks.callbacks import Callbacks
from sklearn import metrics
from indexer import Indexer


parser = NeonArgparser(__doc__)
parser.add_argument('-elec', '--electrode', default=0, help='electrode index')
args = parser.parse_args()
pattern = '*.' + str(args.electrode) + '.wav'
data_dir = args.data_dir
if data_dir[-1] != '/':
    data_dir += '/'
subj = int(data_dir[-2])
assert subj in [1, 2, 3]
indexer = Indexer(data_dir, [args.electrode])
tain_idx, eval_idx, test_idx = indexer.run(data_dir, pattern)

fs = 400
cd = 240000 * 1000 / fs
common_params = dict(sampling_freq=fs, clip_duration=cd, frame_duration=512)
tain_params = AudioParams(**common_params)
eval_params = AudioParams(**common_params)
common = dict(target_size=1, nclasses=2)
test_dir = data_dir.replace('train', 'test')
tain = DataLoader(set_name='tain', media_params=tain_params, index_file=tain_idx,
                  repo_dir=data_dir, shuffle=True, **common)
eval = DataLoader(set_name='eval', media_params=eval_params, index_file=eval_idx,
                  repo_dir=data_dir, **common)
init = Gaussian(scale=0.01)
strides =  dict(str_h=1, str_w=2)
bn = dict(batch_norm=True, activation=Rectlin())
layers = [Conv((3, 5, 64), init=init, activation=Rectlin(), strides=dict(str_h=1, str_w=4)),
          Pooling(2, strides=2),
          Conv((3, 3, 64), init=init, strides=strides, **bn),
          Conv((3, 3, 128), init=init, strides=strides, **bn),
          Conv((3, 3, 128), init=init, strides=strides, **bn),
          Conv((3, 3, 256), init=init, **bn),
          Conv((3, 3, 256), init=init, **bn),
          DeepBiRNN(32, init=GlorotUniform(), reset_cells=True, depth=5, **bn),
          Reshape((1, 64, -1)),
          Conv((1, 3, 256), init=init, strides=dict(str_h=1, str_w=2), **bn),
          Conv((1, 3, 128), init=init, **bn),
          Conv((1, 3, 64), init=init, **bn),
          Conv((1, 3, 32), init=init, **bn),
          Conv((1, 3, 16), init=init, **bn),
          Conv((1, 2, 8), init=init, **bn),
          Affine(nout=common['nclasses'], init=init, activation=Softmax())]

model = Model(layers=layers)
opt = Adagrad(learning_rate=0.01)
callbacks = Callbacks(model, eval_set=eval, **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

model.fit(tain, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
preds = model.get_outputs(eval)[:, 1]
labels = np.loadtxt(eval_idx, delimiter=',', skiprows=1, usecols=[1])
print('Eval AUC for subject %d: %.4f' % (metrics.roc_auc_score(labels, preds), subj))

eval_preds = os.path.join(data_dir, 'eval.' + str(args.electrode) + '.npy')
np.save(eval_preds, preds)

test = DataLoader(set_name='test', media_params=eval_params, index_file=test_idx,
                  repo_dir=test_dir, **common)
preds = model.get_outputs(test)[:, 1]
test_file = 'test.' + str(subj) + '.' + str(args.electrode) + '.npy'
np.save(test_file, preds)
