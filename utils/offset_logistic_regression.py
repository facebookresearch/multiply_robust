# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix, dia_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted


class OffsetLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda_reg=1.0, reg_type="l2"):
        self.reg_type = reg_type
        self.lambda_reg = lambda_reg

    def l2_offset_penalty(self):
        """L2 penalty of the parameters in the model

        Parameters
        ----------

        Returns
        -------
        pen : float
            Returns penalty value
        """
        pen = self.lambda_reg * np.sum(self.parameters**2) / 2
        return pen

    def init_parameters(self, base_margin, X, init_params=None):
        """Initialize the parameters of the offset logistic regression.

        Parameters
        ----------

        base_margin : array-like, shape (n_samples,)
            The values of the logistic(pred) transform of the reference model. These are the offsets and do not have an associated parameter.

        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        None
        """
        d = X.shape[1]
        self.d = d
        self.n = X.shape[0]
        self.base_margin = base_margin
        if init_params is None:
            self.parameters = np.zeros(d)
        else:
            self.parameters = init_params

    def init_sparse_X(self, X):
        """Stores a sparse representation of X for faster computation with scipy when X is sparse

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Attributes
        ----------

        sparse_X: array-like scipy.sparse._coo.coo_matrix
            The training input samples as a sparse matrix

        Returns
        -------
        None

        """
        self.sparse_X = coo_matrix(X)

    def mu_vec(self):
        """Stores a sparse representation of X for faster computation with scipy when X is sparse

        Parameters
        ----------
        None


        Returns
        -------
        pred : array-like, shape (n_samples,)
            The predicted probabilities of class 1 vector of the training data

        """
        pred = 1 / (1 + np.exp(-(self.base_margin + self.sparse_X @ self.parameters)))
        return pred

    def loss(self, X, y, sample_weight=None):
        """Initialize the parameters of the offset logistic regression.

        Parameters
        ----------


        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples, )
            The training input labels.

        sample_weight : array-like, shape (n_samples, )
            The training input weights for each observation.

        Returns
        -------
        loss: float

        """
        mu_vec = self.mu_vec()
        if sample_weight is not None:
            base_loss = -np.sum(sample_weight * y * np.log(mu_vec)) - np.sum(
                sample_weight * (1 - y) * np.log(1 - mu_vec)
            )
            if self.reg_type == "l2":
                loss = base_loss + self.l2_offset_penalty()
                return loss
        else:
            base_loss = -np.mean(y * np.log(mu_vec)) - np.mean(
                (1 - y) * np.log(1 - mu_vec)
            )
            if self.reg_type == "l2":
                loss = base_loss + self.l2_offset_penalty()
                return loss

    def AIC(self, X, y, sample_weight):
        """Compute the AIC for the offset regression model

        Parameters
        ----------


        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples, )
            The training input labels.

        sample_weight : array-like, shape (n_samples, )
            The training input sample_weight for each observation.

        Returns
        -------
        AIC: float

        """
        sp_X = coo_matrix(X)
        train_loss = self.loss(X, y, sample_weight)
        sp_H = coo_matrix(self.hessian(X, y, sample_weight))
        df = np.trace(sp_X.transpose().dot(sp.sparse.linalg.inv(sp_H).dot(sp_X)))
        return train_loss + df

    def gradient(self, X, y, sample_weight=None):
        if sample_weight is not None:
            score = self.sparse_X.transpose().dot(sample_weight * (self.mu_vec() - y))
            if self.reg_type == "l2":
                return score + self.lambda_reg * self.parameters
        else:
            score = (1 / self.n) * self.sparse_X.transpose().dot((self.mu_vec() - y))
            if self.reg_type == "l2":
                return score + self.lambda_reg * self.parameters

    def hessian(self, X, y, sample_weight=None):

        if sample_weight is not None:
            self.sparse_S = sp.sparse.diags(
                sample_weight * self.mu_vec() * (1 - self.mu_vec()), 0
            )
            Q = self.sparse_X.transpose().dot(self.sparse_S.dot(self.sparse_X))
            if self.reg_type == "l2":
                return Q + dia_matrix(np.diag(self.lambda_reg * np.ones(self.d)))
        else:
            self.sparse_S = sp.sparse.diags(self.mu_vec() * (1 - self.mu_vec()), 0)
            Q = (1 / self.n) * self.sparse_X.transpose().dot(
                self.sparse_S.dot(self.sparse_X)
            )
            if self.reg_type == "l2":
                return Q + dia_matrix(np.diag(self.lambda_reg * np.ones(self.d)))

    def newton_step(self, X, y, sample_weight=None):
        H = self.hessian(X, y, sample_weight)
        grad = self.gradient(X, y, sample_weight)
        step = sp.sparse.linalg.spsolve(H, grad)
        self.parameters = self.parameters - step

        return self

    def fit(
        self,
        X,
        y,
        base_margin,
        sample_weight=None,
        init_params=None,
        verbose=False,
        max_iter=50,
        early_stopping=True,
        early_stopping_thresh=1e-8,
    ):
        """Fit the offset regularized logistic regression model

        Parameters
        ----------


        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples, )
            The training input labels.

        base_margin : array-like, shape (n_samples,)
            The values of the logistic(pred) transform of the reference model. These are the offsets and do not have an associated parameter.

        sample_weight : array-like, shape (n_samples, )
            The training input sample_weight for each observation.

        init_params: array_like, shape (n_features, )
            Initial set of parameters for the difference between the base_margin and predictions. The default is all 0s.

        verbose: Logical
            Whether to include messages during optimization

        max_iter: Int
            Maximum number of iterations of the optimization method. Note for l2 regularization, we simply use a Newton method which tends to converge after few iterations.

        early_stopping: Logical
            Whether to stop early once the loss ratio does not change between subsequent iterations.

        early_stopping_thresh: float
            Threshold for improvement on the previous iteration before early stopping.

        Returns
        -------
        self
            The trained model can be accessed for prediction after this iteration.

        """
        # Add intercept
        X_int = np.append(X, np.ones([X.shape[0], 1]), axis=1)

        self.init_parameters(base_margin, X_int, init_params)
        self.init_sparse_X(X_int)

        prev_loss = self.loss(X_int, y, sample_weight)

        if self.reg_type == "l2":
            if verbose:
                print("Starting Offset L2 Regression...")
            for i in range(max_iter):
                self.newton_step(X_int, y, sample_weight)
                stopping_criteria = (
                    self.loss(X_int, y, sample_weight) - prev_loss
                ) / prev_loss
                if (stopping_criteria > -early_stopping_thresh) & early_stopping:
                    if verbose:
                        print("Early Stopping Criteria Reached!")
                    break
                if verbose and i % 1 == 0:
                    print(
                        "\r Iteration {} of {} ::: Loss : {}".format(
                            i + 1,
                            max_iter,
                            np.round(self.loss(X_int, y, sample_weight), 8),
                        )
                    )
                prev_loss = self.loss(X_int, y, sample_weight)
        self.stopping_criteria = stopping_criteria
        self.num_iter = i
        self.is_fitted_ = True

    def predict_proba(self, X, base_margin):
        """Predicts the probabilities of the fine-tuned model given a set of base-model predictions and a set of new features

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            The new input features.

        base_margin : array-like, shape (n_samples,)
            The values of the logistic(pred) transform of the reference model corresponfing to the new features. These do not have a parameter associated with them.

        Returns
        -------
        pred
            Predictions on the new dataset using the base model and the feature array.

        """
        # Only returns probabilities
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        # Add intercept
        X_int = np.append(X, np.ones([X.shape[0], 1]), axis=1)

        pred = 1 / (1 + np.exp(-(base_margin + X_int @ self.parameters)))

        return np.transpose(np.array([1 - pred, pred]))

    def predict(self, X, base_margin, threshold=0.5):
        """Predicts the probability of the fine-tuned model given a set of base-model predictions and a set of new features

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            The new input features.

        base_margin : array-like, shape (n_samples,)
            The values of the logistic(pred) transform of the reference model corresponfing to the new features. These do not have a parameter associated with them.

        Returns
        -------
        pred
            Label predictions
        """
        return 1 * (self.predict_proba(X, base_margin)[:, 1] > threshold)
