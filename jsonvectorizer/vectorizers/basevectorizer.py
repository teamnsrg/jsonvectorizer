from sklearn import base


class BaseVectorizer(base.BaseEstimator, base.TransformerMixin):
    """Base class for vectorizers

    Base class for extracting features from individual fields in JSON
    documents. Any class that inherits from this one must implement a
    scikit-learn-like interface, i.e., :meth:`fit` and :meth:`transform`
    methods. The :meth:`fit` method must accept arbitrary keyword
    arguments, i.e., `**kwargs` at the end of the method's signature,
    and must return None upon failure.

    """

    def fit_transform(self, values, **fit_params):
        """Fit vectorizer to the provided data, then transform it

        Parameters
        ----------
        values : array-like, [n_samples]
            Data for fitting/transforming.
        **fit_params
            Keyword arguments, passed to the :meth:`fit` method.

        Returns
        -------
        X : ndarray, [n_samples, n_features]
            Feature matrix.

        """
        return base.TransformerMixin.fit_transform(self, values, **fit_params)
