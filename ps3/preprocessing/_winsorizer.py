import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Initialize the Winsorizer with the specified lower and upper quantiles.
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Compute the lower and upper quantile thresholds from the data.
        """
        X = np.asarray(X)
        self.lower_quantile_ = np.quantile(X, self.lower_quantile)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile)
        return self

    def transform(self, X):
        """
        Clip the data at the computed quantiles and return the clipped array.
        """
        check_is_fitted(self, ['lower_quantile_', 'upper_quantile_'])
        X = np.asarray(X)
        return np.clip(X, self.lower_quantile_, self.upper_quantile_)
