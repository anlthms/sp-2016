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
    ./model.py -e 16 -w </path/to/data> -r 0 -z 64 -elec <electrode index>
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
from neon.callbacks.callbacks import Callbacks
from sklearn import metrics
from loader import SingleLoader as Loader


parser = NeonArgparser(__doc__)
parser.add_argument('-elec', '--electrode', default=0, help='electrode index')
parser.add_argument('-out', '--out_dir', default='preds', help='directory to write output files')
parser.add_argument('-validate', '--validate_mode', action="store_true", help="validate on training data")

args = parser.parse_args()
elec = args.electrode
data_dir = args.data_dir
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if data_dir[-1] != '/':
    data_dir += '/'
subj = int(data_dir[-2])

tain = Loader(data_dir, subj, elec, args.validate_mode, training=True)
test = Loader(data_dir, subj, elec, args.validate_mode, training=False)

gauss = Gaussian(scale=0.01)
glorot = GlorotUniform()
tiny = dict(str_h=1, str_w=1)
small = dict(str_h=1, str_w=2)
big = dict(str_h=1, str_w=4)
common = dict(batch_norm=True, activation=Rectlin())
layers = [Conv((3, 5, 128), init=gauss, strides=big, **common),
          Pooling(2, strides=2),
          Dropout(0.8),
          Conv((3, 3, 256), init=gauss, strides=small, **common),
          Pooling(2, strides=2),
          Dropout(0.4),
          Conv((3, 3, 512), init=gauss, strides=tiny, **common),
          Dropout(0.2),
          Conv((2, 2, 1024), init=gauss, strides=tiny, **common),
          DeepBiRNN(64, init=glorot, reset_cells=True, depth=3, **common),
          RecurrentMean(),
          Affine(nout=2, init=gauss, activation=Softmax())]

model = Model(layers=layers)
opt = Adagrad(learning_rate=0.0001)
callbacks = Callbacks(model, eval_set=test, **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

model.fit(tain, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
preds = model.get_outputs(test)[:, 1]

if args.validate_mode:
    preds_name = 'eval.'
    idx_file = os.path.join(data_dir, 'eval-' + str(subj) + '-' + str(elec) + '-index.csv')
    labels = np.loadtxt(idx_file, delimiter=',', skiprows=1, usecols=[1])
    auc = metrics.roc_auc_score(labels, preds)
    print('Eval AUC for subject %d: %.4f' % (subj, auc))
else:
    preds_name = 'test.'

preds_file = preds_name + str(subj) + '.' + str(elec) + '.npy'
np.save(os.path.join(out_dir, preds_file), preds)
