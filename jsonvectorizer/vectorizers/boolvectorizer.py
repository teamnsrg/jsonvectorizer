import numpy as np
from sklearn import utils

from .basevectorizer import BaseVectorizer


class BoolVectorizer(BaseVectorizer):
    """Vectorizer for booleans

    Simply creates one feature, i.e., the binarized version of the
    provided array.

    Attributes
    ----------
    feature_names_ : list of str
        Array mapping from feature integer indices to feature names.

    """

    def fit(self, values, **kwargs):
        """Fit vectorizer to the provided data

        Parameters
        ----------
        values : array-like, [n_samples]
            Booleans for fitting the vectorizer.
        **kwargs
            Ignored keyword arguments.

        Returns
        -------
        self or None
            Returns None if `values` only includes one unique boolean
            item, otherwise returns `self`.

        Raises
        ------
        ValueError
            If `values` is not a one-dimensional array.

        """
        values = np.asarray(values, dtype=bool)
        if values.ndim != 1:
            raise ValueError(
                'values must be a one dimensional array, not with shape {}'
                .format(values.shape)
            )

        if np.unique(values).shape[0] == 1:
            return None
        else:
            self.feature_names_ = ['= True']
            return self

    def transform(self, values):
        """Transform booleans to feature matrix

        Parameters
        ----------
        values : array-like, [n_samples]
            Booleans for transforming.

        Returns
        -------
        X : ndarray, [n_samples, 1]
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

        values = np.asarray(values, dtype=bool)
        if values.ndim != 1:
            raise ValueError(
                'values must be a one dimensional array, not with shape {}'
                .format(values.shape)
            )

        return np.expand_dims(values, 1)
