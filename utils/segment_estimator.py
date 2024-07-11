# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class SegmentEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, model, model_segments):

        self.model = model
        self.model_segments = model_segments

    def fit(self, X, y, segments, sample_weight=None):
        if self.model_segments is not None:
            seg_idx = np.isin(segments, self.model_segments)
        else:
            seg_idx = np.arange(len(y))

        if sample_weight is None:
            self.model.fit(X[seg_idx], y[seg_idx])
        else:
            self.model.fit(X[seg_idx], y[seg_idx], sample_weight=sample_weight[seg_idx])

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
