# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.utils.validation import check_array, check_is_fitted


class RegressorLinearCombination(BaseEstimator, ClassifierMixin):
    def __init__(self, model_list, fit_intercept=True, unit_ball=True, verbose=False):
        self.model_list = model_list
        self.fit_intercept = fit_intercept
        self.unit_ball = unit_ball
        self.verbose = verbose
        for model in self.model_list:
            assert hasattr(model, "predict")

    def predict_collection(self, X):
        X = check_array(X)
        y_pred = []
        for model in self.model_list:
            y_pred.append(
                model.predict(X).reshape(-1).astype(np.float64)
            )  
        return np.transpose(np.array(y_pred))

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        max_iters=50,
        C_max=1000,
        C_min=0,
        thresh=10 ** (-4),
    ):
        pred_features = self.predict_collection(X)
        self.LinearRegression = LinearRegression(fit_intercept=self.fit_intercept)
        self.LinearRegression.fit(pred_features, y, sample_weight=sample_weight)

        # constraint to the unit ball, by adjusting the l2 tuning parameter C
        if self.unit_ball:
            coef_norm_sq = np.sum(self.LinearRegression.coef_**2)
            if coef_norm_sq > 1.0:
                if self.verbose:
                    print("Unregularized model satisfies the unit ball constraint")
                next
            # to match the logistic regression notation we use alpha = 1/(2C)
            alpha_min = 1 / (2 * C_max)
            self.LinearRegression = Ridge(alpha=alpha_min)
            self.LinearRegression.fit(pred_features, y, sample_weight=sample_weight)
            coef_norm_sq = np.sum(self.LinearRegression.coef_**2)
            iter_count = 1
            if coef_norm_sq > 1.0:
                diff = np.abs(coef_norm_sq - 1)
                # initialize the bounds
                C_max_tmp = C_max
                C_min_tmp = C_min
                while diff > thresh:
                    C_tmp = (C_max_tmp + C_min_tmp) / 2
                    alpha_tmp = 1 / (2 * C_tmp)
                    self.LinearRegression.set_params(alpha=alpha_tmp)
                    self.LinearRegression.fit(
                        pred_features, y, sample_weight=sample_weight
                    )
                    coef_norm_sq = np.sum(self.LinearRegression.coef_**2)
                    if coef_norm_sq > 1:
                        C_max_tmp = C_tmp
                    else:
                        C_min_tmp = C_tmp
                    diff = np.abs(coef_norm_sq - 1)
                    iter_count += 1
                    if iter_count > max_iters:
                        break
                    if self.verbose:
                        print(
                            "Square norm of parameters ",
                            coef_norm_sq,
                            " C: ",
                            C_tmp,
                            " :: iteration:",
                            iter_count,
                            end="\r",
                        )
            else:
                if self.verbose:
                    print("C_max gives a valid solution")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        pred_features = self.predict_collection(X)
        return self.LinearRegression.predict(pred_features)


class ClassifierLinearCombination(BaseEstimator, ClassifierMixin):
    def __init__(self, model_list, fit_intercept=True, unit_ball=True, verbose=False):
        self.model_list = model_list
        self.fit_intercept = fit_intercept
        for model in self.model_list:
            assert hasattr(model, "predict_proba")
        self.unit_ball = unit_ball
        self.verbose = verbose

    def predict_collection(self, X):
        X = check_array(X)
        model_probs = []

        for model in self.model_list:
            y_pred_prob = model.predict_proba(X)
            if y_pred_prob.shape[1] > 2:
                raise ValueError("More than two classes detected.")
            elif y_pred_prob.shape[1] == 2:
                model_probs.append(y_pred_prob[:, 1])
            else:
                model_probs.append(y_pred_prob)

        return np.transpose(np.array(model_probs))

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        max_iters=50,
        C_max=1000,
        C_min=0,
        thresh=10 ** (-4),
    ):

        pred_features = self.predict_collection(X)
        X_pred_features = np.log(
            pred_features / (1 - pred_features + 10 ** (-15)) + 10 ** (-15)
        )
        self.LogisticRegression = LogisticRegression(
            C=np.inf, fit_intercept=self.fit_intercept
        )  # logistic regression with C=inf is unregularized
        self.LogisticRegression.fit(X_pred_features, y, sample_weight=sample_weight)

        # constraint to the unit ball, by adjusting the l2 tuning parameter C
        if self.unit_ball:
            coef_norm_sq = np.sum(self.LogisticRegression.coef_**2)
            if coef_norm_sq > 1.0:
                if self.verbose:
                    print("Unregularized model satisfies the unit ball constraint")
                next
            self.LogisticRegression.set_params(C=C_max)
            self.LogisticRegression.fit(X_pred_features, y, sample_weight=sample_weight)
            coef_norm_sq = np.sum(self.LogisticRegression.coef_**2)
            iter_count = 1
            if coef_norm_sq > 1.0:
                diff = np.abs(coef_norm_sq - 1)
                # initialize the bounds
                C_max_tmp = C_max
                C_min_tmp = C_min
                while diff > thresh:
                    C_tmp = (C_max_tmp + C_min_tmp) / 2
                    self.LogisticRegression.set_params(C=C_tmp)
                    self.LogisticRegression.fit(
                        X_pred_features, y, sample_weight=sample_weight
                    )
                    coef_norm_sq = np.sum(self.LogisticRegression.coef_**2)
                    if coef_norm_sq > 1:
                        C_max_tmp = C_tmp
                    else:
                        C_min_tmp = C_tmp
                    diff = np.abs(coef_norm_sq - 1)
                    iter_count += 1
                    if iter_count > max_iters:
                        break
                    if self.verbose:
                        print(
                            "Square norm of parameters ",
                            coef_norm_sq,
                            " C: ",
                            C_tmp,
                            " :: iteration:",
                            iter_count,
                            end="\r",
                        )
            else:
                if self.verbose:
                    print("C_max gives a valid solution")
                # raise Warning("C_max is gives a valid solution")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        pred_features = self.predict_collection(X)
        X_pred_features = np.log(
            pred_features / (1 - pred_features + 10 ** (-15)) + 10 ** (-15)
        )
        return self.LogisticRegression.predict(X_pred_features)

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        pred_features = self.predict_collection(X)
        X_pred_features = np.log(
            pred_features / (1 - pred_features + 10 ** (-15)) + 10 ** (-15)
        )
        return self.LogisticRegression.predict_proba(X_pred_features)


class MultiClassifierLinearCombination(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_list,
        model_names=None,
        fit_intercept=True,
        unit_ball=True,
        verbose=False,
        multi_class="multinomial",
    ):
        self.model_list = model_list
        if model_names is not None:
            assert len(model_names) == len(model_list)
            self.model_names = model_names
        else:
            self.model_names = [str(i) for i in range(len(model_list))]
        self.model_with_names = [
            (name, model) for name, model in zip(self.model_names, self.model_list)
        ]
        self.fit_intercept = fit_intercept
        for model in self.model_list:
            assert hasattr(model, "predict_proba")
        self.unit_ball = unit_ball
        self.verbose = verbose
        self.multi_class = multi_class

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        max_iters=50,
        C_max=1000,
        C_min=0,
        thresh=10 ** (-4),
    ):

        self.LogisticRegression = LogisticRegression(
            C=np.inf, fit_intercept=self.fit_intercept, multi_class=self.multi_class
        )  # logistic regression with C=inf is unregularized

        self.stacked_clf = StackingClassifier(
            estimators=self.model_with_names,
            final_estimator=self.LogisticRegression,
            cv="prefit",
            passthrough=False,  # ensures we are fitting only using the outputs of the training set
        )
        self.stacked_clf.fit(X, y, sample_weight=sample_weight)
        self.K = len(self.stacked_clf.classes_)  # number of classes

        # constraint to the unit ball, by adjusting the l2 tuning parameter C
        if self.unit_ball:
            coef_norm_sq = np.sum(self.stacked_clf.final_estimator_.coef_**2)
            if (
                coef_norm_sq > 1.0 * self.K**2
            ):  # for the model stacking, we scale based on the number of classes.
                if self.verbose:
                    print("Unregularized model satisfies the unit ball constraint")
                next

            self.LogisticRegression = LogisticRegression(
                C=C_max, fit_intercept=self.fit_intercept, multi_class=self.multi_class
            )  # reset the regularization parameter

            self.stacked_clf = StackingClassifier(
                estimators=self.model_with_names,
                final_estimator=self.LogisticRegression,
                cv="prefit",
                passthrough=False,
            )
            self.stacked_clf.fit(X, y, sample_weight=sample_weight)
            coef_norm_sq = np.sum(self.stacked_clf.final_estimator_.coef_**2)
            iter_count = 1
            if coef_norm_sq > 1.0 * self.K**2:
                diff = np.abs(coef_norm_sq - 1)
                # initialize the bounds
                C_max_tmp = C_max
                C_min_tmp = C_min
                while diff > thresh:
                    C_tmp = (C_max_tmp + C_min_tmp) / 2

                    self.LogisticRegression = LogisticRegression(
                        C=C_tmp,
                        fit_intercept=self.fit_intercept,
                        multi_class=self.multi_class,
                    )  # reset the regularization parameter

                    self.stacked_clf = StackingClassifier(
                        estimators=self.model_with_names,
                        final_estimator=self.LogisticRegression,
                        cv="prefit",
                        passthrough=False,
                    )

                    self.stacked_clf.fit(X, y, sample_weight=sample_weight)
                    coef_norm_sq = np.sum(self.stacked_clf.final_estimator_.coef_**2)
                    if coef_norm_sq > 1.0 * self.K**2:
                        C_max_tmp = C_tmp
                    else:
                        C_min_tmp = C_tmp
                    diff = np.abs(coef_norm_sq - 1)
                    iter_count += 1
                    if iter_count > max_iters:
                        break
                    if self.verbose:
                        print(
                            "Square norm of parameters ",
                            coef_norm_sq,
                            " C: ",
                            C_tmp,
                            " :: iteration:",
                            iter_count,
                            end="\r",
                        )
            else:
                if self.verbose:
                    print("C_max gives a valid solution")
                # raise Warning("C_max is gives a valid solution")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return self.stacked_clf.predict(X)

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return self.stacked_clf.predict_proba(X)
