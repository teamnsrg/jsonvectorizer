import datetime
import dateutil.parser
import pytz
import scipy.sparse as sp

from .basevectorizer import BaseVectorizer
from .numbervectorizer import NumberVectorizer
from ..utils import _validation


EPOCH = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)


def parse_timestamp(timestamp):
    # Parse a timestamp and return a unix timestamp
    timestamp = dateutil.parser.parse(timestamp)
    if timestamp.tzinfo is None:
        timestamp = pytz.utc.localize(timestamp)

    return (timestamp - EPOCH).total_seconds()


class TimestampVectorizer(BaseVectorizer):
    """Vectorizer for timestamps

    Parses and converts strings to unix timestamps, bins results into
    equiprobable bins, and uses one-hot encoding to create a binary
    feature matrix. After binning, the resulting bins are processed from
    left to right, and are merged into their right neighbor until all
    bins contain at least the specified number of items. If necessary,
    the right-most bin is then merged into its left neighbor. Also, if
    at least `min_f` items are not valid timestamps, an additional bin
    (feature) is created for such items.

    Parameters
    ----------
    n_bins : int
        Number of bins to generate.
    min_f : int or float, optional (default=1)
        Minimum number of samples in each generated bin. An integer is
        taken as an absolute count, and a float indicates the proportion
        of `n_total` passed to the :meth:`fit` method.

    Raises
    ------
    ValueError
        If `n_bins` is not a positive integer, or if `min_f` is not a
        positive numbers.

    Attributes
    ----------
    feature_names_ : list of str
        Array mapping from feature integer indices to feature names.

    """

    def __init__(self, n_bins, min_f=1):
        _validation.check_positive_int(n_bins, alias='n_bins')
        _validation.check_positive(min_f, alias='min_f')
        self.n_bins = n_bins
        self.min_f = min_f

    def fit(self, values, n_total=None, **kwargs):
        """Fit vectorizer to the provided data

        Parameters
        ----------
        values : array-like, [n_samples]
            Timestamps for fitting the vectorizer.
        n_total : int or None, optional (default=None)
            Total Number of documents that values are extracted from. If
            None, defaults to ``len(values)``.
        **kwargs
            Ignored keyword arguments.

        Returns
        -------
        self or None
            Returns `self` if at least two bins are generated, otherwise
            returns None.

        """
        if n_total is None:
            n_total = len(values)

        if isinstance(self.min_f, float):
            min_f = max(int(self.min_f * n_total), 1)
        else:
            min_f = self.min_f

        timestamps = []
        n_invalid = 0
        for value in values:
            try:
                timestamps.append(parse_timestamp(value))
            except (ValueError, OverflowError):
                n_invalid += 1

        has_invalid_feature = (n_invalid >= min_f)
        vectorizer = NumberVectorizer(self.n_bins, min_f=min_f)
        vectorizer = vectorizer.fit(timestamps, n_total=n_total)
        if not has_invalid_feature and vectorizer is None:
            return None

        feature_names = []
        if has_invalid_feature:
            feature_names.append('is not a valid timestamp')
        if vectorizer is not None:
            bin_edges = [
                datetime.datetime.fromtimestamp(bin_edge).isoformat() + 'Z'
                for bin_edge in vectorizer._bin_edges
            ]
            feature_names.append('in (-inf, {})'.format(bin_edges[0]))
            feature_names.extend([
                'in [{}, {})'.format(bin_edges[i], bin_edges[i+1])
                for i in range(len(bin_edges) - 1)
            ])
            feature_names.append('in [{}, inf)'.format(bin_edges[-1]))

        self._has_invalid_feature = has_invalid_feature
        self._vectorizer = vectorizer
        self.feature_names_ = feature_names
        return self

    def transform(self, values):
        """Transform values to feature matrix

        Parameters
        ----------
        values : array-like, [n_samples]
            Timestamps for transforming.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Feature matrix.

        Raises
        ------
        NotFittedError
            If the vectorizer has not yet been fitted.
        ValueError
            If `values` is not a one-dimensional array.

        """
        if not hasattr(self, 'feature_names_'):
            raise NotFittedError('Vectorizer has not yet been fitted')

        invalids = []
        valids = []
        timestamps = []
        for i, value in enumerate(values):
            try:
                timestamps.append(parse_timestamp(value))
                valids.append(i)
            except (ValueError, OverflowError):
                invalids.append(i)

        n_values = len(invalids) + len(valids)
        X = sp.lil_matrix(
            (n_values, len(self.feature_names_)), dtype=bool
        )
        if self._has_invalid_feature:
            if invalids:
                X[invalids,0] = 1
            if valids and self._vectorizer is not None:
                X[valids,1:] = self._vectorizer.transform(timestamps)
        elif valids:
            X[valids,:] = self._vectorizer.transform(timestamps)

        return X
