import numpy as np
import os
import pandas as pd
import pydotplus
import common
from xgboost import XGBClassifier
from os.path import join
from os import remove
from sklearn import tree
from operator import itemgetter
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix


pydotplus.find_graphviz()


def build_model(data_set, data_split, no_interactions, negative, max_snps, cross_validation, output_dir):
    param_grid = {
        'colsample_bytree': [0.6],
        'eta': [0.01, 0.05, 0.1],
        'eval_metric': ['auc'],
        'max_depth':[1, 2, 3, 4, 6],
        'min_child_weight': [1, 2, 3, 4, 5],
        'objective': ['binary:logistic'],
        'subsample': [0.6, 0.8],
        'n_estimators': [500, 1000]
    }

    # Best Estimators parameter grid for Eye color dataset, saves time modeling the entire grid
    # param_grid = {
    #     'colsample_bytree': [0.6],
    #     'eta': [0.01],
    #     'eval_metric': ['auc'],
    #     'max_depth': [1],
    #     'min_child_weight': [5],
    #     'objective': ['binary:logistic'],
    #     'subsample': [0.6],
    #     'n_estimators': [500]
    # }

    model_eval = {
        'roc': get_roc_probs,
        'features': save_features
    }

    model = XGBClassifier()
   
    common.build_model(
        data_set,
        data_split,
        True,
        negative,
        model,
        cross_validation,
        max_snps,
        output_dir,
        param_grid,
        model_eval
    )

def get_roc_probs(model, x_test):
    """
    Gets the prediction probabilities to generate an ROC curve
    :param model: The trained model
    :param x_test: The test data
    :return: The prediction probabilities for the test data
    """
    return model.predict_proba(x_test)

def save_features(model, model_terms, output_dir):
    ftrs = pd.DataFrame()
    ftrs['Feature'] = model_terms
    ftrs['Importance'] = model.feature_importances_
    ftrs.sort_values(by='Importance', ascending=False, inplace=True)
    ftrs.set_index('Feature', inplace=True)
    ftrs.to_csv(os.path.join(output_dir, "xg_features.csv"))
