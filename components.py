from typing import Union, Set

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator


def plot_roc_curve(y_true: S, y_score, feature_name="", ax=None):
    """
    Plot a ROC curve.

    feature_name    optional parameter for plot title (on new axis) or legend label (on existing axis)
    ax              optional parameter enabling the ROC curve to be added to an existing Axes instance.
                    e.g. use f, ax = plt.subplots(1) then pass in ax.
    """

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1

    plt_label = f"AUC {auc:.2f}; Gini {gini:.2f}"
    if ax is None:
        _, ax = plt.subplots(1, figsize=(15, 10))
        ax.set_title(f"{feature_name} ROC curve")
    else:
        plt_label = feature_name + ": " + plt_label

    ax.plot(fpr, tpr, label=plt_label)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    return ax


def classification_report_dict(
    y_true: Union[Series, ndarray],
    y_pred: Union[Series, ndarray],
    y_score: Union[Series, ndarray] = None,
    mirror: bool = False,
    include_ys: bool = True,
    name: str = None,
):
    """
    Return a classification report for a feature as a dict. Useful for bulk reporting on features in a DF.

    y_true      Ground truth
    y_pred      Estimated targets from classifier
    y_score     Classifier target scores or confidences
    mirror      Invert negative Ginis in reporting?
    include_ys  Include `y_pred`, `y_score` and `y_true` in returned dictionary?
    name        Feature name (optional, default `y_pred` column name)
    """

    class_cols = [
        "_".join(x) for x in (product(("precision", "recall", "f1_score", "support"), ("0", "1")))
    ]

    class_report = {"name": name or y_pred.name}

    if include_ys:
        class_report.update({"y_pred": y_pred, "y_score": y_score, "y_true": y_true})

    # Attempt to ensure we have the same set of values in `y_pred` and `y_true`.
    if (len(set(y_pred) - set(y_true)) <= 1 or len(set(y_pred)) <= 2) and (
        y_pred.astype(np.int64) == y_pred
    ).all():
        class_results = precision_recall_fscore_support(y_true, y_pred)
        class_results = [val for pair in class_results for val in pair]
        class_report.update({k: v for k, v in zip(class_cols, class_results)})

    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
            auc = 1 - auc if mirror and auc < 0.5 else auc
            gini = 2 * auc - 1

        except ValueError as exception:
            # Throws if all `y_true` of 1 class.
            auc, gini = None, None
            print("{}: {}".format(name, exception))

        class_report.update({"gini": gini, "auc": auc})

    return class_report


class FilterByHighCorrelation(TransformerMixin, BaseEstimator):
    """
    Compute all pairwise correlations between features. Moving across the DataFrame
    from left to right, we remove the right-most feature of any pair whose pairwise
    correlation is above threshold. Original order (amongst preserved features) is maintained.

    Note: If input features are sorted in descending order of preference, the lower-preference
    feature is always the one removed.
    """

    def __init__(self, threshold: float, output_file: str = None) -> None:
        self.threshold = threshold
        self.output_file = output_file

    def _create_upper_triangle_mask(self, length: tuple) -> np.array:
        mask = np.ones((length, length), dtype="bool")
        mask[np.tril_indices(length)] = False
        return mask

    def _generate_absolute_upper_triangle_corrs(self, corrs: DataFrame, mask: ndarray) -> DataFrame:
        return corrs.abs().where(mask, np.nan)

    def _drop_fts_above_corr_threshold(self, corr_upper_tri: DataFrame, threshold: float) -> list:
        dropped = set()

        for index, row in corr_upper_tri.iterrows():
            if index not in dropped:
                to_drop = set(row.loc[row > threshold].index)
                dropped = dropped.union(to_drop)
        return list(dropped)

    def fit(self, X: DataFrame, y=None):
        self.in_shape = X.shape
        self.corrs = X.corr()

        mask = self._create_upper_triangle_mask(len(self.corrs))
        abs_corr_upper_tri = self._generate_absolute_upper_triangle_corrs(self.corrs, mask)
        self.dropped = self._drop_fts_above_corr_threshold(abs_corr_upper_tri, self.threshold)
        self.kept = [ft for ft in X.columns if ft not in self.dropped]
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        return X.loc[:, self.kept]


def label_encode(data: DataFrame) -> DataFrame:
    """
    Encode labels so we can use them in the model.

    Some of the preprocessing code here is from this
    [introductory notebook](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction/notebook).
    """

    encoder = LabelEncoder()
    le_count = 0
    le_cols = []

    # Iterate through the columns
    for col in data:
        if data[col].dtype == "object":
            # If 2 or fewer unique categories
            if len(list(data[col].unique())) <= 2:
                # Train on the training data
                encoder.fit(data[col])
                # Transform both training and testing data
                data[col] = encoder.transform(data[col])
                le_cols.append(col)

                # Keep track of how many columns were label encoded
                le_count += 1

    print(f"{le_count} columns were label encoded. These were {le_cols}")
    return data
