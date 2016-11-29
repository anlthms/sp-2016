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
Utility functions.
"""
import numpy as np
from sklearn import metrics
from prep import nwin


def avg(labels, preds):
    assert preds.shape[0] % nwin == 0
    preds_len = preds.shape[0] // nwin
    post_preds = np.zeros(preds_len, np.float32)
    post_labels = np.zeros(preds_len, np.float32)
    for i in range(preds_len):
        post_preds[i] = np.mean(preds[nwin*i:nwin*(i+1)])
        post_labels[i] = labels[nwin*i]
        assert post_labels[i] == np.mean(labels[nwin*i:nwin*(i+1)])
    return (post_labels, post_preds)


def avg_preds(preds):
    assert preds.shape[0] % nwin == 0
    preds_len = preds.shape[0] // nwin
    post_preds = np.zeros(preds_len, np.float32)
    for i in range(preds_len):
        post_preds[i] = np.mean(preds[nwin*i:nwin*(i+1)])
    return post_preds


def auc(labels, preds):
    return metrics.roc_auc_score(labels, preds)


def score(labels, preds):
    return auc(*avg(labels, preds))
