# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from xgboost import XGBClassifier

from ..utils.offset_logistic_regression import OffsetLogisticRegression

from .linear_combinations import (
    ClassifierLinearCombination,
    MultiClassifierLinearCombination,
    RegressorLinearCombination,
)


class MultiplyRobust(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator_list,
        weights_generator,
        residual_estimator,
        residual_estimator_set=None,
        segmented_shift=False,
        verbose=True,
        fit_intercept=True,
        unit_ball=True,
        center_weights=True,
    ):
        self.fit_intercept = fit_intercept
        self.unit_ball = unit_ball
        self.base_estimator_list = base_estimator_list
        self.first_estimator = RegressorLinearCombination(
            self.base_estimator_list,
            fit_intercept=self.fit_intercept,
            unit_ball=self.unit_ball,
        )

        self.weights_generator = weights_generator
        self.residual_estimator = residual_estimator
        if residual_estimator_set is None:
            self.residual_estimator_set = {}
        else:
            self.residual_estimator_set = residual_estimator_set
        self.segmented_shift = segmented_shift
        self.verbose = verbose
        self.first_estimator_set = {}
        self.center_weights = center_weights

    def fit(
        self,
        X,
        y,
        X_test,
        X_weights=None,
        X_weights_test=None,
        segments=None,
        segments_test=None,
        segment_set=None,
        idx_holdout=None,
    ):
        X, y = check_X_y(X, y)

        # get domain shift weights
        if (X_weights is None) or (X_weights_test is None):
            w = self.weights_generator.compute_weights(X, X_test, y)
        else:
            w = self.weights_generator.compute_weights(X_weights, X_weights_test, y)

        n = len(y)
        idx = np.arange(n)
        if idx_holdout is None:
            self.idx_holdout = idx  # learn the stage 1 model on the whole set. Can give overfitting problem if same training data is used.
        else:
            self.idx_holdout = idx_holdout

        # fit the same iteration on all of the segments
        if segments is not None:
            if segment_set is None:
                segment_set = np.unique(segments)
            for seg in segment_set:
                if self.verbose:
                    print(seg, end="\r")
                idx_seg = np.where(segments == seg)[0]
                y_seg = y[idx_seg]
                X_seg = X[idx_seg, :]
                w_seg = w[idx_seg]

                if idx_holdout is not None:
                    idx_seg_holdout = np.intersect1d(idx_seg, idx_holdout)
                    y_seg_holdout = y[idx_seg_holdout]
                    X_seg_holdout = X[idx_seg_holdout, :]

                else:
                    y_seg_holdout = y_seg
                    X_seg_holdout = X_seg

                if segments_test is not None:
                    idx_test_seg = np.where(segments == seg)[0]
                    X_test_seg = X_test[idx_test_seg, :]

                self.first_estimator_set[seg] = copy.deepcopy(self.first_estimator)

                # default estimator for the second stage if we do not have a single one
                if seg in list(self.residual_estimator_set.keys()):
                    self.residual_estimator_set[seg] = self.residual_estimator_set[seg]
                else:
                    self.residual_estimator_set[seg] = copy.deepcopy(
                        self.residual_estimator
                    )

                self.first_estimator_set[seg].fit(X_seg_holdout, y_seg_holdout)
                y_seg_base_pred = self.first_estimator_set[seg].predict(X_seg)

                # step2: get domain shift weights if they are in fact segmented
                if self.segmented_shift:
                    w_seg = self.weights_generator.compute_weights(
                        X_seg, X_test_seg, y_seg
                    )

                w_seg = w_seg / np.sum(w_seg)

                if self.center_weights:
                    w_seg = w_seg / np.mean(w_seg)

                # step3: fit residual
                y_seg_res = y_seg - y_seg_base_pred
                self.residual_estimator_set[seg].fit(X_seg, y_seg_res, w_seg)
        else:
            raise ValueError("segments is None")
        self.trained_segments = list(self.residual_estimator_set.keys())
        self.is_fitted_ = True
        return self

    def predict(self, X, segments=None):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        if segments is None:
            y_pred = self.first_estimator.predict(X) + self.residual_estimator.predict(
                X
            )

        else:
            n_pred = X.shape[0]
            y_pred = np.repeat(0.0, n_pred)

            pred_segment_set = np.unique(segments)
            for seg in pred_segment_set:
                idx_seg = np.where(segments == seg)[0]
                X_seg = X[idx_seg, :]
                if seg in self.trained_segments:
                    y_pred_seg = self.first_estimator_set[seg].predict(
                        X_seg
                    ) + self.residual_estimator_set[seg].predict(X_seg)
                else:
                    y_pred_seg = self.first_estimator.predict(
                        X_seg
                    ) + self.residual_estimator.predict(X_seg)
                y_pred[idx_seg] = y_pred_seg

        return y_pred


class MultiplyRobustClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator_list,
        weights_generator,
        second_estimator,
        second_estimator_set=None,
        segmented_shift=False,
        verbose=True,
        fit_intercept=True,
        unit_ball=True,
        center_weights=True,
    ):

        if (type(second_estimator) is type(OffsetLogisticRegression())) & (
            type(second_estimator) is type(XGBClassifier())
        ):
            raise NotImplementedError("!!!")
        if type(second_estimator) is type(OffsetLogisticRegression()):
            self.second_model_type = "lr"
        if type(second_estimator) is type(XGBClassifier()):
            self.second_model_type = "xgb"
        self.center_weights = center_weights
        self.fit_intercept = fit_intercept
        self.unit_ball = unit_ball
        self.base_estimator_list = base_estimator_list
        self.first_estimator = ClassifierLinearCombination(
            self.base_estimator_list,
            fit_intercept=self.fit_intercept,
            unit_ball=self.unit_ball,
        )
        self.weights_generator = weights_generator
        self.second_estimator = second_estimator
        if second_estimator_set is None:
            self.second_estimator_set = {}
        else:
            self.second_estimator_set = second_estimator_set
        self.segmented_shift = segmented_shift
        self.verbose = verbose
        self.first_estimator_set = {}  # linear combinations

    def fit(
        self,
        X,
        y,
        X_test,
        X_weights=None,
        X_weights_test=None,
        segments=None,
        segment_set=None,
        segments_test=None,
        idx_holdout=None,  # holdout set for first stage linear combination
    ):
        X, y = check_X_y(X, y)

        # to reduce the complexity of the function, we do not include the unsegmented version here.

        # get domain shift weights
        if (X_weights is None) or (X_weights_test is None):
            w = self.weights_generator.compute_weights(X, X_test, y)
        else:
            w = self.weights_generator.compute_weights(X_weights, X_weights_test, y)

        n = len(y)
        idx = np.arange(n)
        if idx_holdout is None:
            self.idx_holdout = idx  # learn the stage 1 model on the whole set. Can give overfitting problem if same training data is used.
        else:
            self.idx_holdout = idx_holdout

        if segments is not None:
            if segment_set is None:
                segment_set = np.unique(segments)
            for seg in segment_set:
                if self.verbose:
                    print(seg, end="\r")
                idx_seg = np.where(segments == seg)[0]

                y_seg = y[idx_seg]
                X_seg = X[idx_seg, :]
                w_seg = w[idx_seg]

                idx_seg_holdout = np.intersect1d(idx_seg, self.idx_holdout)
                y_seg_holdout = y[idx_seg_holdout]
                X_seg_holdout = X[idx_seg_holdout, :]

                if segments_test is not None:
                    idx_test_seg = np.where(segments == seg)[0]
                    X_test_seg = X_test[idx_test_seg, :]

                # Using data splitting estimators if indicated.
                self.first_estimator_set[seg] = ClassifierLinearCombination(
                    self.base_estimator_list,
                    fit_intercept=self.fit_intercept,
                    unit_ball=self.unit_ball,
                )

                # default estimator for the second stage if we do not have a single one
                if seg not in list(self.second_estimator_set.keys()):
                    self.second_estimator_set[seg] = copy.deepcopy(
                        self.second_estimator
                    )

                # Using data splitting estimators if indicated.
                self.first_estimator_set[seg].fit(X_seg_holdout, y_seg_holdout)

                y_pred_seg = self.first_estimator_set[seg].predict_proba(X_seg)[:, 1]
                base_m_seg = np.log(
                    y_pred_seg / (1 - y_pred_seg + 10 ** (-15)) + 10 ** (-15)
                )

                # step2: get domain shift weights if they are in fact segmented
                if self.segmented_shift:
                    w_seg = self.weights_generator.compute_weights(
                        X_seg, X_test_seg, y_seg
                    )
                w_seg = w_seg / np.sum(w_seg)
                # step3: fit second model, different syntax for logistic regression and xgb

                if (self.second_model_type == "xgb") | (self.center_weights):
                    w_seg = (
                        w_seg / w_seg.mean()
                    )  # un-normalize when the second stage model is xgb

                if self.second_model_type in ["lr", "xgb"]:
                    self.second_estimator_set[seg].fit(
                        X_seg, y_seg, base_margin=base_m_seg, sample_weight=w_seg
                    )

        self.trained_segments = list(self.second_estimator_set.keys())
        self.is_fitted_ = True
        return self

    def predict_proba(self, X, segments=None):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        if segments is None:
            raise NotImplementedError("!!!")

        else:
            n_pred = X.shape[0]
            y_pred_proba = np.repeat(np.nan, n_pred)
            pred_segment_set = np.unique(segments)
            for seg in pred_segment_set:
                idx_seg = np.where(segments == seg)[0]
                X_seg = X[idx_seg, :]
                y_pred_seg = self.first_estimator_set[seg].predict_proba(X_seg)[:, 1]
                base_m_seg = np.log(
                    y_pred_seg / (1 - y_pred_seg + 10 ** (-15)) + 10 ** (-15)
                )

                if self.second_model_type in ["lr", "xgb"]:
                    y_pred_proba_seg = self.second_estimator_set[seg].predict_proba(
                        X_seg, base_margin=base_m_seg
                    )[:, 1]

                y_pred_proba[idx_seg] = y_pred_proba_seg

        return y_pred_proba

    def predict(self, X, segments=None, threshold=0.5):
        return 1 * (self.predict_proba(X, segments) > threshold)

    def first_model_predict_proba(self, X, segments=None):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        if segments is None:
            raise NotImplementedError("!!!")
        else:
            n_pred = X.shape[0]
            y_pred_proba = np.repeat(0.0, n_pred)
            pred_segment_set = np.unique(segments)
            for seg in pred_segment_set:
                idx_seg = np.where(segments == seg)[0]
                X_seg = X[idx_seg, :]
                y_pred_proba_seg = self.first_estimator_set[seg].predict_proba(X_seg)[
                    :, 1
                ]
                y_pred_proba[idx_seg] = y_pred_proba_seg

        return y_pred_proba


class MultiplyRobustMultiClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator_list,
        weights_generator,
        second_estimator,
        num_class,
        second_estimator_set=None,
        segmented_shift=False,
        verbose=True,
        fit_intercept=True,
        unit_ball=True,
        multi_class="multinomial",
        center_weights=True,
    ):

        self.multi_class = multi_class
        self.num_class = num_class
        self.center_weights = center_weights
        self.fit_intercept = fit_intercept
        self.unit_ball = unit_ball
        self.base_estimator_list = base_estimator_list
        self.first_estimator = MultiClassifierLinearCombination(
            self.base_estimator_list,
            fit_intercept=self.fit_intercept,
            unit_ball=self.unit_ball,
        )
        self.weights_generator = weights_generator
        self.second_estimator = second_estimator
        if second_estimator_set is None:
            self.second_estimator_set = {}
        else:
            self.second_estimator_set = second_estimator_set
        self.segmented_shift = segmented_shift
        self.verbose = verbose
        self.first_estimator_set = {}  # linear combinations

    def fit(
        self,
        X,
        y,
        X_test,
        X_weights=None,
        X_weights_test=None,
        segments=None,
        segment_set=None,
        segments_test=None,
        idx_holdout=None,  # holdout set for first stage linear combination
    ):
        X, y = check_X_y(X, y)

        # to reduce the complexity of the function, we do not include the unsegmented version here.

        # get domain shift weights
        if (X_weights is None) or (X_weights_test is None):
            w = self.weights_generator.compute_weights(X, X_test, y)
        else:
            w = self.weights_generator.compute_weights(X_weights, X_weights_test, y)

        n = len(y)
        idx = np.arange(n)
        if idx_holdout is None:
            self.idx_holdout = idx  # learn the stage 1 model on the whole set. Can give overfitting problem if same training data is used.
        else:
            self.idx_holdout = idx_holdout

        if segments is not None:
            if segment_set is None:
                segment_set = np.unique(segments)
            for seg in segment_set:
                if self.verbose:
                    print(seg, end="\r")
                idx_seg = np.where(segments == seg)[0]

                y_seg = y[idx_seg]
                X_seg = X[idx_seg, :]
                w_seg = w[idx_seg]

                idx_seg_holdout = np.intersect1d(idx_seg, self.idx_holdout)
                y_seg_holdout = y[idx_seg_holdout]
                X_seg_holdout = X[idx_seg_holdout, :]

                if segments_test is not None:
                    idx_test_seg = np.where(segments == seg)[0]
                    X_test_seg = X_test[idx_test_seg, :]

                # Using data splitting estimators if indicated.
                self.first_estimator_set[seg] = MultiClassifierLinearCombination(
                    self.base_estimator_list,
                    fit_intercept=self.fit_intercept,
                    unit_ball=self.unit_ball,
                    multi_class=self.multi_class,
                )

                # default estimator for the second stage if we do not have a single one
                if seg not in list(self.second_estimator_set.keys()):
                    self.second_estimator_set[seg] = copy.deepcopy(
                        self.second_estimator
                    )

                # Using data splitting estimators if indicated.
                self.first_estimator_set[seg].fit(X_seg_holdout, y_seg_holdout)

                y_pred_seg = self.first_estimator_set[seg].predict_proba(X_seg)
                base_m_seg = np.log(y_pred_seg + 10 ** (-15))

                # step2: get domain shift weights if they are in fact segmented
                if self.segmented_shift:
                    w_seg = self.weights_generator.compute_weights(
                        X_seg, X_test_seg, y_seg
                    )
                w_seg = w_seg / np.sum(w_seg)
                # step3: fit second model, different syntax for logistic regression and xgb

                if self.center_weights:
                    w_seg = (
                        w_seg / w_seg.mean()
                    )  # un-normalize when the second stage model is xgb

                self.second_estimator_set[seg].fit(
                    X_seg,
                    y_seg,
                    base_margin=base_m_seg.flatten(),
                    sample_weight=w_seg,
                )

        self.trained_segments = list(self.second_estimator_set.keys())
        self.is_fitted_ = True
        return self

    def predict_proba(self, X, segments=None):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        if segments is None:
            raise NotImplementedError("!!!")

        else:
            n_pred = X.shape[0]
            y_pred_proba = np.nan * np.zeros((n_pred, self.num_class))
            pred_segment_set = np.unique(segments)
            for seg in pred_segment_set:
                idx_seg = np.where(segments == seg)[0]
                X_seg = X[idx_seg, :]
                y_pred_seg = self.first_estimator_set[seg].predict_proba(X_seg)
                base_m_seg = np.log(y_pred_seg + 10 ** (-15))

                y_pred_proba_seg = self.second_estimator_set[seg].predict_proba(
                    X_seg, base_margin=base_m_seg.flatten()
                )

                y_pred_proba[idx_seg, :] = y_pred_proba_seg

        return y_pred_proba

    def predict(self, X, segments=None):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        if segments is None:
            raise NotImplementedError("!!!")

        else:
            n_pred = X.shape[0]
            y_pred = np.nan * np.zeros(n_pred)
            pred_segment_set = np.unique(segments)
            for seg in pred_segment_set:
                idx_seg = np.where(segments == seg)[0]
                X_seg = X[idx_seg, :]
                y_pred_seg_stage1 = self.first_estimator_set[seg].predict_proba(X_seg)
                base_m_seg = np.log(y_pred_seg_stage1 + 10 ** (-15))
                y_pred_seg = self.second_estimator_set[seg].predict(
                    X_seg, base_margin=base_m_seg.flatten()
                )

                y_pred[idx_seg] = y_pred_seg

        return y_pred
