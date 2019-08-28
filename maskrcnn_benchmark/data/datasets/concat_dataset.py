# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect

import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def __init__(self, datasets, uniform_datasets):
        _ConcatDataset.__init__(self, datasets)

        self.uniform_datasets = uniform_datasets

    def get_idxs(self, idx):
        if self.uniform_datasets:
            dataset_idx = np.random.randint(len(self.cumulative_sizes))
            if dataset_idx == 0:
                low = 0
            else:
                low = self.cumulative_sizes[dataset_idx - 1]
            sample_idx = np.random.randint(0, self.cumulative_sizes[dataset_idx] - low)
        else:
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        dataset_idx, sample_idx = self.get_idxs(idx)

        return self.datasets[dataset_idx][sample_idx]
