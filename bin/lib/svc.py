from sklearn.svm import SVC
import pandas as pd

class SVCF(SVC):
    """ SVC Classifier with features """

    def __init__(self, all_features=None, features=None, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):

        """ Initialise """

        self.all_features = all_features
        self.features = features

        super().__init__(C, kernel, degree, gamma,
                     coef0, shrinking, probability,
                     tol, cache_size, class_weight,
                     verbose, max_iter, decision_function_shape,
                     random_state)

    def fit(self, X, y):
        """ Select parameters and fit """

        if self.all_features is None or self.features is None:
            raise "features or all_features not set. Add all_features and features in construction and/or param_grid"

        X_sel = pd.DataFrame(X, columns=self.all_features).loc[:,self.features].values
        return super().fit(X_sel, y)

    def predict(self, X):
        """ Select parameters and predict """

        if self.all_features is None or self.features is None:
            raise "features or all_features not set. Add all_features and features in construction and/or param_grid"

        X_sel = pd.DataFrame(X, columns=self.all_features).loc[:,self.features].values
        return super().predict(X_sel)

    def predict_proba(self, X):
        """ Select parameters and predict with probabilities"""

        if self.all_features is None or self.features is None:
            raise "features or all_features not set. Add all_features and features in construction and/or param_grid"
        

        X_sel = pd.DataFrame(X, columns=self.all_features).loc[:,self.features].values
        return super().predict_proba(X_sel)
