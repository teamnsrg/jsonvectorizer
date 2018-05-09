import copy
import numpy as np
from sklearn import feature_extraction, preprocessing, utils

from .basevectorizer import BaseVectorizer
from ..utils import _validation


class StringVectorizer(BaseVectorizer):
    """Vectorizer for strings

    Converts strings to an arrays of binary features. If the number of
    unique values during fitting is small (less than `max_categories`
    and ``len(values) / n_average``), uses categorical features with
    one-hot encoding, otherwise uses tokenization (using scikit-learn's
    :class:`CountVectorizer`). If exactly two unique values (categories)
    are seen, only generates one feature.

    Parameters
    ----------
    max_categories : int, optional (default=1)
        Maximum number of categories for categorical features.
    n_average : int or float, optional (default=1)
        Average number of samples in each category for categorical
        features. An integer is taken as an absolute count, and a float
        indicates the proportion of `n_total` passed to the :meth:`fit`
        method.
    min_df : int or float, optional (default=1)
        When using tokenization, ignore terms that have a document
        frequency strictly lower than this threshold. An integer is
        taken as an absolute count, and a float indicates the proportion
        of `n_total` passed to the :meth:`fit` method.
    **kwargs
        Passed to scikit-learn's :class:`CountVectorizer` class for
        initialization.

    Raises
    ------
    ValueError
        If `max_categories` is not a positive integer, or if `n_average`
        is not a positive number.

    Attributes
    ----------
    feature_names_ : list of str
        Array mapping from feature integer indices to feature names.

    """

    def __init__(
        self, max_categories=1, n_average=1, min_df=1, **kwargs
    ):
        _validation.check_positive_int(max_categories, alias='max_categories')
        _validation.check_positive(n_average, alias='n_average')

        self.max_categories = max_categories
        self.n_average = n_average
        self.params = dict(min_df=min_df, **kwargs)

    def fit(self, values, n_total=None, **kwargs):
        """Fit vectorizer to the provided data

        Parameters
        ----------
        values : array-like, [n_samples]
            Strings for fitting the vectorizer.
        n_total : int or None, optional (default=None)
            Total Number of documents that values are extracted from. If
            None, defaults to ``len(values)``.
        **kwargs:
            Ignored Keyword arguments.

        Returns
        -------
        self or None
            Returns None if `values` only includes one unique item,
            otherwise returns `self`.

        """
        values = [value.lower() for value in values]
        if n_total is None:
            n_total = len(values)

        if isinstance(self.n_average, float):
            n_average = self.n_average * n_total
        else:
            n_average = float(self.n_average)

        params = copy.copy(self.params)
        if isinstance(params['min_df'], float):
            params['min_df'] = max(int(params['min_df'] * n_total), 1)
        else:
            params['min_df'] = params['min_df']

        unique_values = set(values)
        max_categories = min(self.max_categories, len(values) / n_average)
        if len(unique_values) <= 1:
            return None
        elif len(unique_values) <= max_categories:
            # Categorization
            self._categorical = True
            self._vectorizer = preprocessing.LabelBinarizer(sparse_output=True)
            self._vectorizer.fit(values)
            if self._vectorizer.y_type_ == 'binary':
                self.feature_names_ = [
                    u'= {1} (!= {0})'.format(*self._vectorizer.classes_)
                ]
            else:
                self.feature_names_ = [
                    u'= {}'.format(category)
                    for category in self._vectorizer.classes_
                ]
        else:
            # Tokenization
            self._categorical = False
            self._vectorizer = feature_extraction.text.CountVectorizer(
                binary=True, dtype=bool, **params
            )
            try:
                self._vectorizer.fit(values)
            except ValueError:
                return None

            self.feature_names_ = [
                u'has token "{}"'.format(feature_name)
                for feature_name in self._vectorizer.get_feature_names()
            ]
            if hasattr(self._vectorizer, 'stop_words_'):
                delattr(self._vectorizer, 'stop_words_')

        return self

    def transform(self, values):
        """Transform values to feature matrix

        Parameters
        ----------
        values : array-like, [n_samples]
            Strings for transforming.

        Returns
        -------
        X : sparse matrix, shape [n_samples, n_features]
            Feature matrix.

        Raises
        ------
        NotFittedError
            If the vectorizer has not yet been fitted.
        ValueError
            If `values` is not a one-dimensional array.

        """
        if not hasattr(self, 'feature_names_'):
            raise utils.NotFittedError('Vectorizer has be yet been fitted')

        if self._categorical:
            values = [value.lower() for value in values]
            return self._vectorizer.transform(values).astype(bool)
        else:
            return self._vectorizer.transform(values)
