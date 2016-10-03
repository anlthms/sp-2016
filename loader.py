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
Load data for all the electrodes.
"""
import numpy as np
from neon.data import DataLoader, AudioParams, NervanaDataIterator
from indexer import Indexer
from prep import ds_factor


class EegLoader(NervanaDataIterator):

    def __init__(self, repo_dir, subj, validate_mode, training):
        # Sampling frequency
        fs = 400 // ds_factor
        # Clip duration in milliseconds
        cd = 10 * 60 * 1000
        self.nelecs = nelecs = 16
        common_params = dict(sampling_freq=fs, clip_duration=cd, frame_duration=512)
        if training:
            media_params = AudioParams(random_scale_percent=5.0, **common_params)
            set_name = 'tain' if validate_mode else 'full'
            data_dir = repo_dir
        else:
            media_params = AudioParams(**common_params)
            set_name = 'eval' if validate_mode else 'test'
            data_dir = repo_dir if validate_mode else repo_dir.replace('train', 'test')
        common = dict(target_size=1, nclasses=2)
        indexer = Indexer(repo_dir, subj, validate_mode, training)
        self.loaders = []
        for elec in range(nelecs):
            index_file = indexer.run(elec)
            loader = DataLoader(set_name=set_name + '-' + str(subj) + '-' + str(elec),
                                media_params=media_params, index_file=index_file,
                                repo_dir=data_dir, **common)
            self.loaders.append(loader)
        self.shape_list = list(media_params.get_shape())
        self.shape_list[0] = nelecs
        self.shape = tuple(self.shape_list)
        datum_size = nelecs * media_params.datum_size()
        self.data = self.be.iobuf(datum_size, dtype=np.float32)
        self.data_shape = (nelecs, self.data.shape[0] // nelecs, -1)
        self.data_view = self.data.reshape(self.data_shape)
        self.ndata = self.loaders[0].ndata
        self.start_idx = self.loaders[0].start_idx

    def start(self):
        map(DataLoader.start, self.loaders)

    def stop(self):
        map(DataLoader.stop, self.loaders)

    def reset(self):
        map(DataLoader.reset, self.loaders)

    @property
    def nbatches(self):
        return self.loaders[0].nbatches

    def next(self, start):
        for elec in range(self.nelecs):
            self.data_view[elec], targets = self.loaders[elec].next(start)
        return self.data, targets

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.be.bsz):
            yield self.next(start)
