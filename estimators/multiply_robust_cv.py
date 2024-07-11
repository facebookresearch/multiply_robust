# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, mean_squared_error

from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    ParameterGrid,
    ShuffleSplit,
    StratifiedGroupKFold,
)

from sklearn.utils.validation import check_X_y

from xgboost import XGBClassifier

from ..utils.offset_logistic_regression import OffsetLogisticRegression

from .linear_combinations import (
    ClassifierLinearCombination,
    MultiClassifierLinearCombination,
    RegressorLinearCombination,
)
from .multiply_robust import (
    MultiplyRobust, 
    MultiplyRobustClassifier, 
    MultiplyRobustMultiClassifier,
)


class MultiplyRobustCV(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator_list,
        weights_generator,
        second_estimator,
        param_grid=None,
        segmented_shift=False,
        verbose=0,
        n_splits=5,
        random_state=42,
        fit_intercept=True,
        unit_ball=True,
        loss=None,
    ):

        # treat the problem as a classifier for the second estimator when it is of these types, otherwise it is a regressor.
        if type(second_estimator) is type(OffsetLogisticRegression()):
            self.second_model_type = "lr"
            self.is_classifier = True
        elif type(second_estimator) is type(XGBClassifier()):
            self.second_model_type = "xgb"
            self.is_classifier = True
        else:
            self.is_classifier = False

        self.fit_intercept = fit_intercept
        self.unit_ball = unit_ball
        self.base_estimator_list = base_estimator_list
        self.base_estimator_list_clone_ = copy.deepcopy(base_estimator_list)
        if self.is_classifier:
            self.first_estimator = ClassifierLinearCombination(
                self.base_estimator_list,
                fit_intercept=self.fit_intercept,
                unit_ball=self.unit_ball,
            )
        else:
            self.first_estimator = RegressorLinearCombination(
                self.base_estimator_list,
                fit_intercept=self.fit_intercept,
                unit_ball=self.unit_ball,
            )
        self.first_estimator_clone_ = copy.deepcopy(self.first_estimator)
        self.weights_generator = weights_generator
        self.second_estimator = second_estimator
        self.second_estimator_clone_ = copy.deepcopy(second_estimator)
        if param_grid is None:
            self.param_grid = {}
        else:
            self.param_grid = param_grid
        self.long_param_grid = ParameterGrid(param_grid)

        self.segmented_shift = segmented_shift
        self.verbose = verbose
        self.n_splits = n_splits
        self.random_state = random_state
        if loss is None:
            if self.is_classifier:
                self.loss = log_loss
            else:
                self.loss = mean_squared_error
        else:
            self.loss = loss

    def k_fold_and_shuffle_split(self, data_split, groups=None):
        if groups is not None:
            self.group_shuffle = True
        else:
            self.group_shuffle = False

        if self.group_shuffle:
            kf = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            ss = GroupShuffleSplit(
                n_splits=self.n_splits,
                random_state=self.random_state,
                test_size=data_split,
            )
        else:
            kf = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            ss = ShuffleSplit(
                n_splits=self.n_splits,
                random_state=self.random_state,
                test_size=data_split,
            )
        return kf, ss

    def fit(
        self,
        X,
        y,
        X_test,
        groups=None,
        segments=None,
        X_weights=None,
        X_weights_test=None,
        data_split=0.2,
        segments_test=None,
    ):
        X, y = check_X_y(X, y)

        # precompute domain shift weights
        if (X_weights is None) or (X_weights_test is None):
            w = self.weights_generator.compute_weights(X, X_test, y)
        else:
            w = self.weights_generator.compute_weights(X_weights, X_weights_test, y)

        # dictionary of arrays for the results
        results_segments = {}
        unique_segments = np.unique(segments)
        for seg in unique_segments:
            results_segments[seg] = np.zeros((len(self.long_param_grid), self.n_splits))

        kf, ss = self.k_fold_and_shuffle_split(data_split, groups)

        for k, (train, val) in enumerate(kf.split(X, y, groups=groups)):
            if self.verbose:
                print("Fold ", k + 1, " of ", self.n_splits, " fitting ...", end="\n")
                print("", end="\n")

            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            w_val = w[val]
            seg_train, seg_val = segments[train], segments[val]
            groups_train = groups[train]

            (idx_base_train, idx_holdout) = next(
                ss.split(X_train, y_train, groups=groups_train)
            )  #  data splitting for the holdout set.

            # train the base models on the base indices
            for model in self.base_estimator_list_clone_:
                model.fit(
                    X_train[idx_base_train],
                    y_train[idx_base_train],
                    segments=seg_train[idx_base_train],
                )

            for i, params in zip(
                range(len(self.long_param_grid)), self.long_param_grid
            ):
                if self.verbose:
                    print(
                        "Parameter set ",
                        i + 1,
                        " of ",
                        len(self.long_param_grid),
                        " fitting ...",
                        end="\r",
                    )
                self.estimator = copy.deepcopy(
                    self.second_estimator_clone_.set_params(**params)
                )

                if self.is_classifier:
                    mr_model = MultiplyRobustClassifier(
                        base_estimator_list=self.base_estimator_list_clone_,
                        weights_generator=self.weights_generator,
                        second_estimator=self.estimator,
                        segmented_shift=self.segmented_shift,
                        verbose=0,
                        fit_intercept=self.fit_intercept,
                        unit_ball=self.unit_ball,
                    )
                else:
                    mr_model = MultiplyRobust(
                        base_estimator_list=self.base_estimator_list_clone_,
                        weights_generator=self.weights_generator,
                        second_estimator=self.estimator,
                        segmented_shift=self.segmented_shift,
                        verbose=0,
                        fit_intercept=self.fit_intercept,
                        unit_ball=self.unit_ball,
                    )
                mr_model.fit(
                    X_train,
                    y_train,
                    X_test,
                    segments=seg_train,
                    idx_holdout=idx_holdout,
                )

                if self.is_classifier:
                    y_pred = mr_model.predict_proba(X_val, segments=seg_val)
                else:
                    y_pred = mr_model.predict(X_val, segments=seg_val)

                for seg in unique_segments:
                    idx_val_seg = np.where(seg_val == seg)[0]
                    loss_seg = self.loss(
                        y_val[idx_val_seg],
                        y_pred[idx_val_seg],
                        sample_weight=w_val[idx_val_seg],
                    )
                    results_segments[seg][i, k] = loss_seg

        self.results_segments = results_segments
        self.best_params_ = {}
        self.best_score_ = {}
        self.results_test_mean = {}
        for seg in unique_segments:
            # we use the mean to select the best model parameters
            results_test_mean = np.mean(results_segments[seg], axis=1)
            self.results_test_mean[seg] = results_test_mean
            i_opt = np.argmin(results_test_mean)

            self.best_params_[seg] = self.long_param_grid[i_opt]
            print("Best parameters segment: ", seg, " :: ", self.best_params_[seg])
            self.best_score_[seg] = np.min(results_test_mean)


class MultiplyRobustMulticlassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator_list,
        weights_generator,
        second_estimator,
        num_class,
        param_grid=None,
        segmented_shift=False,
        verbose=0,
        n_splits=5,
        random_state=42,
        fit_intercept=True,
        unit_ball=False,
        multi_class="multinomial",
        loss=log_loss,
    ):

        self.multi_class = self.multi_class
        self.loss = loss
        self.num_class = num_class
        self.fit_intercept = fit_intercept
        self.unit_ball = unit_ball
        self.base_estimator_list = base_estimator_list
        self.base_estimator_list_clone_ = copy.deepcopy(base_estimator_list)
        self.first_estimator = MultiClassifierLinearCombination(
            self.base_estimator_list,
            fit_intercept=self.fit_intercept,
            unit_ball=self.unit_ball,
            multi_class=self.multi_class,
        )
        self.first_estimator_clone_ = copy.deepcopy(self.first_estimator)
        self.weights_generator = weights_generator
        self.second_estimator = second_estimator
        self.second_estimator_clone_ = copy.deepcopy(second_estimator)
        if param_grid is None:
            self.param_grid = {}
        else:
            self.param_grid = param_grid
        self.long_param_grid = ParameterGrid(param_grid)

        self.segmented_shift = segmented_shift
        self.verbose = verbose
        self.n_splits = n_splits
        self.random_state = random_state

        if self.is_classifier:
            self.loss = log_loss
        else:
            self.loss = mean_squared_error

    def k_fold_and_shuffle_split(self, data_split, groups=None):
        if groups is not None:
            self.group_shuffle = True
        else:
            self.group_shuffle = False

        if self.group_shuffle:
            kf = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            ss = GroupShuffleSplit(
                n_splits=self.n_splits,
                random_state=self.random_state,
                test_size=data_split,
            )
        else:
            kf = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            ss = ShuffleSplit(
                n_splits=self.n_splits,
                random_state=self.random_state,
                test_size=data_split,
            )
        return kf, ss

    def fit(
        self,
        X,
        y,
        X_test,
        groups=None,
        segments=None,
        X_weights=None,
        X_weights_test=None,
        data_split=0.2,
        segments_test=None,
    ):
        X, y = check_X_y(X, y)

        # precompute domain shift weights
        if (X_weights is None) or (X_weights_test is None):
            w = self.weights_generator.compute_weights(X, X_test, y)
        else:
            w = self.weights_generator.compute_weights(X_weights, X_weights_test, y)

        # dictionary of arrays for the results
        results_segments = {}
        unique_segments = np.unique(segments)
        for seg in unique_segments:
            results_segments[seg] = np.zeros((len(self.long_param_grid), self.n_splits))

        kf, ss = self.k_fold_and_shuffle_split(data_split, groups)

        for k, (train, val) in enumerate(kf.split(X, y, groups=groups)):
            if self.verbose:
                print("Fold ", k + 1, " of ", self.n_splits, " fitting ...", end="\n")
                print("", end="\n")

            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            w_val = w[val]
            seg_train, seg_val = segments[train], segments[val]
            groups_train = groups[train]

            (idx_base_train, idx_holdout) = next(
                ss.split(X_train, y_train, groups=groups_train)
            )  #  data splitting for the holdout set.

            # train the base models on the base indices
            for model in self.base_estimator_list_clone_:
                model.fit(
                    X_train[idx_base_train],
                    y_train[idx_base_train],
                    segments=seg_train[idx_base_train],
                )

            for i, params in zip(
                range(len(self.long_param_grid)), self.long_param_grid
            ):
                if self.verbose:
                    print(
                        "Parameter set ",
                        i + 1,
                        " of ",
                        len(self.long_param_grid),
                        " fitting ...",
                        end="\r",
                    )
                self.estimator = copy.deepcopy(
                    self.second_estimator_clone_.set_params(**params)
                )

                mr_model = MultiplyRobustMultiClassifier(
                    base_estimator_list=self.base_estimator_list_clone_,
                    weights_generator=self.weights_generator,
                    second_estimator=self.estimator,
                    num_class=self.num_class,
                    segmented_shift=self.segmented_shift,
                    verbose=0,
                    fit_intercept=self.fit_intercept,
                    unit_ball=self.unit_ball,
                    multi_class=self.multi_class,
                )

                mr_model.fit(
                    X_train,
                    y_train,
                    X_test,
                    segments=seg_train,
                    idx_holdout=idx_holdout,
                )

                y_pred_proba = mr_model.predict_proba(X_val, segments=seg_val)

                for seg in unique_segments:
                    idx_val_seg = np.where(seg_val == seg)[0]
                    loss_seg = self.loss(
                        y_val[idx_val_seg],
                        y_pred_proba[idx_val_seg],
                        sample_weight=w_val[idx_val_seg],
                    )
                    results_segments[seg][i, k] = loss_seg

        self.results_segments = results_segments
        self.best_params_ = {}
        self.best_score_ = {}
        self.results_test_mean = {}
        for seg in unique_segments:
            # we use the mean to select the best model parameters
            results_test_mean = np.mean(results_segments[seg], axis=1)
            self.results_test_mean[seg] = results_test_mean
            i_opt = np.argmin(results_test_mean)

            self.best_params_[seg] = self.long_param_grid[i_opt]
            print("Best parameters segment: ", seg, " :: ", self.best_params_[seg])
            self.best_score_[seg] = np.min(results_test_mean)
