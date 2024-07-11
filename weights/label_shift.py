# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

class LabelShiftWeights:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def compute_weights(self, X, X_test, y):

        y_pred_prob = self.model.predict_proba(X)
        K = y_pred_prob.shape[1]

        # if a method is used which does not give the exact class, then this will threshold them to the appropriate class.
        y_pred = self.model.predict(X).astype("int")
        y_pred_test = self.model.predict(X_test).astype("int")

        n = len(y)  # number of training samples
        n_test = len(y_pred_test)  # number of testing samples

        C = np.zeros([K, K])
        mu_test = np.zeros([K])
        for k in range(K):
            for j in range(K):
                C[j, k] = np.sum(1 * (y == k) * (y_pred == j))
        C = C / n

        for k in range(K):
            mu_test[k] = np.sum(y_pred_test == k)
        mu_test = mu_test / n_test
        C_inv = np.linalg.inv(C)
        train_class_weights = C_inv @ mu_test

        if min(train_class_weights) < 0:
            raise Exception("Negative weights are not allowed")

        w = train_class_weights[y]
        # normalize the weights
        w = w / sum(w)

        return w
