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
Train a per-subject model for:
https://www.kaggle.com/c/melbourne-university-seizure-prediction

Usage:
    ./model.py -w </path/to/data> -r 0 -z 64 -elec <electrode index or -1>
"""

import os
import sys
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian, GlorotUniform
from neon.layers import Conv, Pooling, GeneralizedCost, Affine
from neon.layers import DeepBiRNN, RecurrentMean, Dropout
from neon.optimizers import Adagrad
from neon.transforms import Rectlin, Softmax, CrossEntropyBinary
from neon.models import Model
from neon.callbacks.callbacks import Callback, Callbacks
from neon import logger
from sklearn import metrics
from util import score


class Evaluator(Callback):
    def __init__(self, subj, data_dir, eval_set):
        super(Evaluator, self).__init__()
        self.subj = subj
        self.data_dir = data_dir
        self.eval_set = eval_set

    def on_epoch_end(self, callback_data, model, epoch):
        preds = model.get_outputs(self.eval_set)[:, 1]
        idx_file = os.path.join(self.data_dir, 'eval-' + str(self.subj) + '-' + str(0) + '-index.csv')
        labels = np.loadtxt(idx_file, delimiter=',', skiprows=1, usecols=[1])
        logger.display('Eval AUC for subject %d epoch %d: %.4f\n' % (self.subj, epoch, score(labels, preds)))

parser = NeonArgparser(__doc__)
parser.add_argument('-elec', '--electrode', default=0, help='electrode index')
parser.add_argument('-out', '--out_dir', default='preds', help='directory to write output files')
parser.add_argument('-validate', '--validate_mode', action="store_true", help="validate on training data")

args = parser.parse_args()
data_dir = os.path.normpath(args.data_dir)
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
subj = int(data_dir[-1])

rate = 0.00001
nepochs = {1: 8, 2: 3, 3: 6}[subj]
logger.warn('Overriding --epochs option')

if args.electrode == '-1':
    from loader import MultiLoader as Loader
    elecs = range(16)
else:
    from loader import SingleLoader as Loader
    rate /= 10
    elecs = args.electrode

tain = Loader(data_dir, subj, elecs, args.validate_mode, training=True)
test = Loader(data_dir, subj, elecs, args.validate_mode, training=False)

gauss = Gaussian(scale=0.01)
glorot = GlorotUniform()
tiny = dict(str_h=1, str_w=1)
small = dict(str_h=1, str_w=2)
big = dict(str_h=1, str_w=4)
common = dict(batch_norm=True, activation=Rectlin())
layers = {1: [Conv((3, 5, 64), init=gauss, strides=big, **common),
          Pooling(2, strides=2),
          Conv((3, 3, 128), init=gauss, strides=small, **common),
          Pooling(2, strides=2),
          Conv((3, 3, 256), init=gauss, strides=small, **common),
          Conv((2, 2, 512), init=gauss, strides=tiny, **common),
          Conv((2, 2, 128), init=gauss, strides=tiny, **common),
          DeepBiRNN(64, init=glorot, reset_cells=True, depth=3, **common),
          RecurrentMean(),
          Affine(nout=2, init=gauss, activation=Softmax())],
          2: [Conv((3, 5, 64), init=gauss, strides=big, **common),
          Pooling(2, strides=2),
          Dropout(0.8),
          Conv((3, 3, 128), init=gauss, strides=small, **common),
          Pooling(2, strides=2),
          Dropout(0.4),
          Conv((3, 3, 256), init=gauss, strides=small, **common),
          Dropout(0.2),
          Conv((2, 2, 512), init=gauss, strides=tiny, **common),
          Conv((2, 2, 128), init=gauss, strides=tiny, **common),
          DeepBiRNN(64, init=glorot, reset_cells=True, depth=5, **common),
          RecurrentMean(),
          Affine(nout=2, init=gauss, activation=Softmax())],
          3: [Conv((3, 5, 64), init=gauss, strides=big, **common),
          Pooling(2, strides=2),
          Dropout(0.8),
          Conv((3, 3, 128), init=gauss, strides=small, **common),
          Pooling(2, strides=2),
          Dropout(0.4),
          Conv((3, 3, 256), init=gauss, strides=small, **common),
          Dropout(0.2),
          Conv((2, 2, 512), init=gauss, strides=tiny, **common),
          Conv((2, 2, 128), init=gauss, strides=tiny, **common),
          DeepBiRNN(64, init=glorot, reset_cells=True, depth=5, **common),
          RecurrentMean(),
          Affine(nout=2, init=gauss, activation=Softmax())]}[subj]


model = Model(layers=layers)
opt = Adagrad(learning_rate=rate)
callbacks = Callbacks(model, eval_set=test, **args.callback_args)
if args.validate_mode:
    evaluator = Evaluator(subj, data_dir, test)
    callbacks.add_callback(evaluator)
    preds_name = 'eval.'
else:
    preds_name = 'test.'
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

model.fit(tain, optimizer=opt, num_epochs=nepochs, cost=cost, callbacks=callbacks)
preds = model.get_outputs(test)[:, 1]
preds_file = preds_name + str(subj) + '.npy'
np.save(os.path.join(out_dir, preds_file), preds)
