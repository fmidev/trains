import logging
import numpy as np

class State():
    SCALER_FITTED = {'classifier_x': False,
                     'classifier_y': False,
                     'regressor_x': False,
                     'regressor_y': False
                     }
    xscaler_classifier = None
    xscaler_regressor = None
    yscaler_regressor = None

    preds = {}

    def add_pred(self, y_pred_proba, y_pred, y, name):
        """ Add predictions and true values """

        if name in self.preds.keys():
            if y_pred_proba is not None:
                self.preds[name]['y_pred_proba'] = np.append(self.preds[name]['y_pred_proba'], y_pred_proba, axis=0)
            self.preds[name]['y_pred'] = np.append(self.preds[name]['y_pred'], y_pred, axis=0)
            self.preds[name]['y'] = np.append(self.preds[name]['y'], y, axis=0)
        else:
            self.preds[name] = {}
            self.preds[name]['y_pred_proba'] = y_pred_proba
            self.preds[name]['y_pred'] = y_pred
            self.preds[name]['y'] = y

    def get_pred(self, name):
        """ Get predictions and true values """
        return self.preds[name]['y_pred_proba'],self.preds[name]['y_pred'],self.preds[name]['y']

    def get_pred_names(self):
        """ Return names of predictions"""
        return self.preds.keys()        
