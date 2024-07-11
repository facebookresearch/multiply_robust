# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage

# Takes a list of domains and compares the distance between each of them
class DomainCluster:
    def __init__(self):
        self.distances_fit_ = False

    def compute_distances(
        self,
        X=None,
        y=None,
        s=None,
        kernel="Gaussian",
        kernel_y="Delta",
        B=1,
        verbose=False,
    ):
        self.kernel = kernel
        self.kernel_y = kernel_y
        self.B = B  # bandwidth parameter
        self.s_values = np.unique(s)
        self.S = len(self.s_values)
        self.verbose = verbose

        if self.kernel != "Gaussian":
            raise NotImplementedError("Only Gaussian kernel has been implemented")

        if self.kernel == "Gaussian":
            self.ker_fun = lambda x: np.exp(
                -(x / (2 * self.B**2))
            )  # takes in the norm value

        if self.kernel_y == "Delta":
            self.ker_fun_y = lambda x: 1 - np.minimum(
                x, 1
            )  # takes in the difference between the two labels

        if self.kernel_y == "Gaussian":
            self.ker_fun_y = lambda x: np.exp(
                -(x / (2 * self.B**2))
            )  # takes in the norm value for the labels when y is continuous

        D = np.zeros((self.S, self.S))
        for j, seg1 in zip(range(self.S), self.s_values):
            for k, seg2 in zip(range(self.S), self.s_values):
                if k > j:
                    if self.verbose:
                        print(f"Computing distance between {seg1} and {seg2}", end="\r")
                    idx1 = np.where(s == seg1)[0]
                    idx2 = np.where(s == seg2)[0]
                    m1 = len(idx1)
                    m2 = len(idx2)

                    X1 = X[idx1, :]
                    X2 = X[idx2, :]

                    y1 = y[idx1]
                    y2 = y[idx2]

                    N1 = np.sum(X1**2, axis=1)
                    N2 = np.sum(X2**2, axis=1)
                    N12 = X1 @ np.transpose(X2)

                    Gy1 = self.ker_fun_y(np.abs(np.subtract.outer(y1, y1)))
                    Gy2 = self.ker_fun_y(np.abs(np.subtract.outer(y2, y2)))
                    Gy12 = self.ker_fun_y(np.abs(np.subtract.outer(y1, y2)))

                    G1 = self.ker_fun(np.add.outer(N1, N1) - 2 * X1 @ np.transpose(X1))
                    G2 = self.ker_fun(np.add.outer(N2, N2) - 2 * X2 @ np.transpose(X2))
                    G12 = self.ker_fun(np.add.outer(N1, N2) - 2 * N12)

                    np.fill_diagonal(G1, 0)
                    np.fill_diagonal(G2, 0)

                    D[j, k] = np.sqrt(
                        np.max(
                            [
                                (1 / (m1 * (m1 - 1))) * np.sum(G1 * Gy1)
                                + (1 / (m2 * (m2 - 1))) * np.sum(G2 * Gy2)
                                - 2 / (m1 * m2) * np.sum(G12 * Gy12),
                                0,
                            ]
                        )
                    )
        self.D = D + D.T
        self.distances_fit_ = True

        return self

    def cluster_domains(self, n_clusters, linkage_type="single"):
        if self.distances_fit_:
            labels = ["domain_" + str(i) for i in self.s_values]
            Z = linkage(self.D, linkage_type)
            return (cut_tree(Z, n_clusters=n_clusters), labels)
        else:
            raise AttributeError(
                "Must compute the distances matrix prior to clustering the domains"
            )
        return self

    def plot_dendrogram(self, linkage_type="single", color_threshold=None):
        if color_threshold is None:
            color_threshold = np.percentile(self.D, 50)

        if self.distances_fit_:
            labels = ["domain_" + str(i) for i in self.s_values]
            Z = linkage(self.D, linkage_type)

            # Visualize the clustering as a dendogram
            plt.figure(figsize=(25, 10))
            dendrogram(
                Z, orientation="right", labels=labels, color_threshold=color_threshold
            )
            plt.show()
            return self
        else:
            raise AttributeError(
                "Must compute the distances matrix prior to clustering the domains"
            )
