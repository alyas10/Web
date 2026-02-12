# model_utils.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

class NumericFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=100):
        self.k = k
        self.selector = None
        self.numeric_columns = None

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.numeric_columns = X.columns.tolist()
        else:
            self.numeric_columns = list(range(X.shape[1]))

        if hasattr(X, 'loc'):
            X_numeric = X[self.numeric_columns]
        else:
            X_numeric = X

        k = min(self.k, X_numeric.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.selector.fit(X_numeric, y)
        return self

    def transform(self, X):
        if hasattr(X, 'loc'):
            X_numeric = X[self.numeric_columns]
            X_transformed = self.selector.transform(X_numeric)
            selected_columns = np.array(self.numeric_columns)[self.selector.get_support()]
            return X_numeric.__class__(X_transformed, columns=selected_columns, index=X.index)
        else:
            return self.selector.transform(X)