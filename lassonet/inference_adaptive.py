import itertools
import sys
import warnings
from abc import ABCMeta, abstractmethod, abstractstaticmethod
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import List

import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.model_selection import check_cv, train_test_split
from tqdm import tqdm

from .cox import CoxPHLoss, concordance_index
from .interfaces import BaseLassoNet, HistoryItem
from .model import LassoNet


class AdaptiveLassoNet(BaseLassoNet):
    def __init__(self, penalty_type='scad', **kwargs):
        super().__init__(**kwargs)
        self.penalty_weights = None
        self.penalty_type = penalty_type

    @staticmethod
    def scad_penalty(z, lamb, a=3.7):
        # First order derivative of SCAD penalty
        # tuning parameter "a" (or "gamma") use the default value 3.7
        if lamb == torch.inf:
            return 1
        return lamb * (1 * (z<=lamb) + max( (a * lamb - z), 0) / ((a - 1) * lamb) * (lamb<z))

    def _train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        batch_size,
        epochs,
        lambda_,
        optimizer,
        return_state_dict,
        patience=None,
    ) -> HistoryItem:
        model = self.model
        if self.penalty_weights is None:
            self.penalty_weights = torch.tensor([1.0] * X_train.shape[-1])
        def validation_obj():
            with torch.no_grad():
                return (
                    self.criterion(model(X_val), y_val).item()
                    + lambda_ * model.l1_regularization_weighted_skip(self.penalty_weights).item()
                    + self.gamma * model.l2_regularization().item()
                    + self.gamma_skip * model.l2_regularization_skip().item()
                )

        best_val_obj = validation_obj()
        epochs_since_best_val_obj = 0
        if self.backtrack:
            best_state_dict = self.model.cpu_state_dict()
            real_best_val_obj = best_val_obj
            real_loss = float("nan")  # if epochs == 0

        n_iters = 0

        n_train = len(X_train)
        if batch_size is None:
            batch_size = n_train
            randperm = torch.arange
        else:
            randperm = torch.randperm
        batch_size = min(batch_size, n_train)

        for epoch in range(epochs):
            indices = randperm(n_train)
            model.train()
            loss = 0
            for i in range(n_train // batch_size):
                # don't take batches that are not full
                batch = indices[i * batch_size : (i + 1) * batch_size]

                def closure():
                    nonlocal loss
                    optimizer.zero_grad()
                    crit = self.criterion(model(X_train[batch]), y_train[batch])
                    ans = (
                        crit
                        + self.gamma * model.l2_regularization()
                        + self.gamma_skip * model.l2_regularization_skip()
                    )
                    if not torch.isfinite(ans):
                        print(f"Loss is {ans}", file=sys.stderr)
                        print("Did you normalize input?", file=sys.stderr)
                        print("Loss::", crit.item())
                        print(
                            "l2_regularization:",
                            model.l2_regularization(),
                        )
                        print(
                            "l2_regularization_skip:",
                            model.l2_regularization_skip(),
                        )
                        assert False
                    ans.backward()
                    loss += ans.item() * batch_size / n_train
                    return ans

                optimizer.step(closure)
                model.prox(
                    lambda_=lambda_ * optimizer.param_groups[0]["lr"] * self.penalty_weights, #.repeat(self.M + 1, 1),
                    M=self.M,
                )

            if epoch == 0:
                # fallback to running loss of first epoch
                real_loss = loss
            model.eval()
            val_obj = validation_obj()
            if val_obj < self.tol * best_val_obj:
                best_val_obj = val_obj
                epochs_since_best_val_obj = 0
            else:
                epochs_since_best_val_obj += 1
            if self.backtrack and val_obj < real_best_val_obj:
                best_state_dict = self.model.cpu_state_dict()
                real_best_val_obj = val_obj
                real_loss = loss
                n_iters = epoch + 1
            if patience is not None and epochs_since_best_val_obj == patience:
                break

        if self.backtrack:
            self.model.load_state_dict(best_state_dict)
            val_obj = real_best_val_obj
            loss = real_loss
        else:
            n_iters = epoch + 1
        with torch.no_grad():
            reg = self.model.l1_regularization_weighted_skip(self.penalty_weights).item()
            l2_regularization = self.model.l2_regularization()
            l2_regularization_skip = self.model.l2_regularization_skip()
        return HistoryItem(
            lambda_=lambda_,
            state_dict=self.model.cpu_state_dict() if return_state_dict else None,
            objective=loss + lambda_ * reg,
            loss=loss,
            val_objective=val_obj,
            val_loss=val_obj - lambda_ * reg,  # TODO remove l2 reg
            regularization=reg,
            l2_regularization=l2_regularization,
            l2_regularization_skip=l2_regularization_skip,
            selected=self.model.input_mask().cpu(),
            n_iters=n_iters,
        )

    def path(
        self,
        X,
        y,
        *,
        X_val=None,
        y_val=None,
        lambda_seq=None,
        lambda_max=float("inf"),
        return_state_dicts=False,
        callback=None,
        disable_lambda_warning=False,
    ) -> List[HistoryItem]:
        """Train LassoNet on a lambda\\_ path.
        The path is defined by the class parameters:
        start at `lambda_start` and increment according to `path_multiplier`.
        The path will stop when no feature is being used anymore.
        callback will be called at each step on (model, history)
        """
        assert (X_val is None) == (
            y_val is None
        ), "You must specify both or none of X_val and y_val"
        sample_val = self.val_size != 0 and X_val is None
        if sample_val:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_size, random_state=self.random_state
            )
        elif X_val is None:
            X_train, y_train = X_val, y_val = X, y
        else:
            X_train, y_train = X, y
        X_train, y_train = self._cast_input(X_train, y_train)
        X_val, y_val = self._cast_input(X_val, y_val)

        hist: List[HistoryItem] = []

        # always init model
        self._init_model(X_train, y_train)
        if self.n_iters_init:
            hist.append(
                self._train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    batch_size=self.batch_size,
                    lambda_=0,
                    epochs=self.n_iters_init,
                    optimizer=self.optim_init(self.model.parameters()),
                    patience=self.patience_init,
                    return_state_dict=return_state_dicts,
                )
            )
        if callback is not None:
            callback(self, hist)
        if self.verbose > 1:
            print("Initialized dense model")
            hist[-1].log()

        optimizer = self.optim_path(self.model.parameters())

        # build lambda_seq
        if lambda_seq is not None:
            pass
        elif self.lambda_seq is not None:
            lambda_seq = self.lambda_seq
        else:

            def _lambda_seq(start):
                while start <= lambda_max:
                    yield start
                    start *= self.path_multiplier

            if self.lambda_start == "auto":
                # divide by 10 for initial training
                self.lambda_start_ = (
                    self.model.lambda_start(M=self.M)
                    / optimizer.param_groups[0]["lr"]
                    / 10
                )
                if self.verbose > 1:
                    print(f"lambda_start = {self.lambda_start_:.2e}")
                lambda_seq = _lambda_seq(self.lambda_start_)
            else:
                lambda_seq = _lambda_seq(self.lambda_start)

        # extract first value of lambda_seq
        lambda_seq = iter(lambda_seq)
        lambda_start = next(lambda_seq)

        is_dense = True
        for current_lambda in itertools.chain([lambda_start], lambda_seq):
            if self.model.selected_count() == 0:
                break
            last = self._train(  # LLA Step 1
                X_train,
                y_train,
                X_val,
                y_val,
                batch_size=self.batch_size,
                lambda_=current_lambda,
                epochs=self.n_iters_path,
                optimizer=optimizer,
                patience=self.patience_path,
                return_state_dict=return_state_dicts,
            )
            skip_weight = self.model.skip.weight.data.abs().squeeze()
            self.penalty_weights = torch.empty(skip_weight.shape[0])
            n = len(X_train)
            lambda0 = current_lambda / n
            for j in range(skip_weight.shape[0]):
                self.penalty_weights[j] = self.scad_penalty(skip_weight[j], lambda0) / lambda0
            last = self._train(  # LLA Step 2
                X_train,
                y_train,
                X_val,
                y_val,
                batch_size=self.batch_size,
                lambda_=current_lambda,
                epochs=self.n_iters_path,
                optimizer=optimizer,
                patience=self.patience_path,
                return_state_dict=return_state_dicts,
            )
            if is_dense and self.model.selected_count() < X_train.shape[1]:
                is_dense = False
                if not disable_lambda_warning and current_lambda < 2 * lambda_start:
                    warnings.warn(
                        f"lambda_start={lambda_start:.3f} "
                        f"{'(selected automatically) ' * (self.lambda_start == 'auto')}"
                        "might be too large.\n"
                        f"Features start to disappear at {current_lambda=:.3f}."
                    )

            hist.append(last)
            if callback is not None:
                callback(self, hist)

            if self.verbose > 1:
                print(
                    f"Lambda = {current_lambda:.2e}, "
                    f"selected {self.model.selected_count()} features "
                )
                last.log()

        self.feature_importances_ = self._compute_feature_importances(hist)
        """When does each feature disappear on the path?"""

        return hist


class AdaptiveLassoNetRegressor(
    RegressorMixin,
    MultiOutputMixin,
    AdaptiveLassoNet,
):
    """Use LassoNet as regressor"""

    def _convert_y(self, y):
        y = torch.FloatTensor(y).to(self.device)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        return y

    @staticmethod
    def _output_shape(y):
        return y.shape[1]

    criterion = torch.nn.MSELoss(reduction="mean")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            ans = self.model(self._cast_input(X))
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans


class AdaptiveLassoNetClassifier(
    ClassifierMixin,
    AdaptiveLassoNet,
):
    """Use LassoNet as classifier

    Parameters
    ----------
    class_weight : iterable of float, default=None
        If specified, weights for different classes in training.
        There must be one number per class.
    """

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def __init__(self, class_weight=None, **kwargs):
        BaseLassoNet.__init__(self, **kwargs)

        self.class_weight = class_weight

        if class_weight is not None:
            self.class_weight = torch.FloatTensor(self.class_weight).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=self.class_weight, reduction="mean"
            )

    __init__.__doc__ = BaseLassoNet.__init__.__doc__

    def _init_model(self, X, y):
        output_shape = self._output_shape(y)
        if self.class_weight is not None:
            assert output_shape == len(self.class_weight)

        return super()._init_model(X, y)

    def _convert_y(self, y) -> torch.TensorType:
        y = torch.LongTensor(y).to(self.device)
        assert len(y.shape) == 1, "y must be 1D"
        return y

    @staticmethod
    def _output_shape(y):
        return (y.max() + 1).item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            ans = self.model(self._cast_input(X)).argmax(dim=1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            ans = torch.softmax(self.model(self._cast_input(X)), -1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans
