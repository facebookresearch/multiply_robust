# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics import pairwise


class KMMWeights:
    """
    Implementation of the Kernel Mean Matching algorithm for covariate shift
    based on

    Huang, J., Gretton, A., Borgwardt, K., Schölkopf, B. and Smola, A., 2006.
    Correcting sample selection bias by unlabeled data. Advances in neural
    information processing systems, 19.

    The code is adapted from the package adapt:
    https://adapt-python.github.io/adapt/index.html

    Parameters:
    -----------
    kernel: Kernel metric.
    Possible values: 
    See https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.pairwise.pairwise_kernels.html for more details.

    B: Upper bound for weights .

    eps: Constraint parameter.

    tol: Optimization threshold. If "None", default parameters from cvxopt are
    used.

    max_iter: Maximal iteration of the optimization.

    kernel_params: Dictionary with additional parameters for the kernel.

    gamma, coef0, degree:
    See https://scikit-learn.org/stable/modules/metrics.html#metrics for more
    details.

    verbose: Print out the progress of the optimization.
    """

    def __init__(
        self,
        kernel="rbf",
        B=1000,  # follow the setting in Section 4.2 in the paper
        eps=None,
        tol=None,
        max_iter=100,
        kernel_params=None,
        verbose=True,
    ):
        self.kernel = kernel
        self.B = B
        self.eps = eps
        self.tol = tol
        self.max_iter = max_iter
        if kernel_params is None:
            self.kernel_params = {}
        else:
            self.kernel_params = kernel_params
        self.verbose = verbose

    def compute_weights(self, X, X_test, y=None):

        n = len(X)
        n_test = len(X_test)

        if self.eps is None:
            # follow the setting in Section 4.2 in the paper
            self.eps = 1.0 - 1.0 / np.sqrt(n)

        K = pairwise.pairwise_kernels(X, X, metric=self.kernel, **self.kernel_params)
        K = 0.5 * (K + K.transpose())  # make it symmetric

        kappa = pairwise.pairwise_kernels(
            X, X_test, metric=self.kernel, **self.kernel_params
        )
        kappa = (n / n_test) * np.dot(kappa, np.ones((n_test, 1)))

        # follow the notations in quadratic programming formulation from cvxopt
        # see https://cvxopt.org/userguide/coneprog.html?highlight=cvxopt%20solvers%20qp
        # for more details
        P = matrix(K)
        q = -matrix(kappa)

        G = np.ones((2 * n + 2, n))
        G[1] = -G[1]
        G[2 : n + 2] = np.eye(n)
        G[n + 2 : n * 2 + 2] = -np.eye(n)
        h = np.ones(2 * n + 2)
        h[0] = n * (1.0 + self.eps)
        h[1] = n * (self.eps - 1.0)
        h[2 : n + 2] = self.B
        h[n + 2 :] = 0

        G = matrix(G)
        h = matrix(h)

        solvers.options["show_progress"] = self.verbose
        solvers.options["maxiters"] = self.max_iter
        if self.tol is not None:
            solvers.options["abstol"] = self.tol
            solvers.options["reltol"] = self.tol
            solvers.options["feastol"] = self.tol
        else:
            # default settings in cvxopt
            solvers.options["abstol"] = 1e-7
            solvers.options["reltol"] = 1e-6
            solvers.options["feastol"] = 1e-7

        weights = np.array(solvers.qp(P, q, G, h)["x"]).ravel()
        return weights / weights.sum()
