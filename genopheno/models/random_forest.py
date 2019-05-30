from models import common
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier


def build_model(dataset, data_split, no_interactions, negative, max_snps, cross_validation, output_dir):
    model_eval = {
        'features': save_features
    }

    default_grid = {
        'n_estimators': [1000],
        'criterion': ['gini'],
        'max_features': ['sqrt'],
        'max_depth': [None],
        'min_samples_leaf': [0.0025],
        'min_samples_split': [0.01]
    }

    # For testing combinations of parameters
    param_grid = {
        "criterion": ["gini", "entropy"],
        # If float then min_samples_split is a percentage and ceil(min_samples_split * n_samples)
        # are the minimum number of samples for each split.
        "min_samples_split": [.01, .015, .02, .025],
        "max_depth": [None, 4, 5],  # int or None. None allows a full tree
        # If float then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples)
        #  are the minimum number of samples for each leaf.
        "min_samples_leaf": [.0025, .005, .01, .015],  # Berry and Linoff .0025 to .01
        # If float then max_features is a percentage and int(min_max_features* n_features)
        # features are considered at each split. If 'auto' then max_features = sqrt(n_features)
        # If None then max_features = n_features
        "max_features": ["sqrt", .3, .4, .5],
        # A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)is 30 to 40%
        "n_estimators": [500, 1000, 3000]
    }

    common.build_model(
        dataset,
        data_split,
        True,
        negative,
        RandomForestClassifier(n_jobs=-1),
        cross_validation,
        max_snps,
        output_dir,
        param_grid=default_grid,
        model_eval=model_eval
    )


def save_features(model, model_terms, output_dir):
    # rf default features
    ftrs = pd.DataFrame()
    ftrs['Feature'] = model_terms
    ftrs['Importance'] = model.feature_importances_
    ftrs.sort_values(by='Importance', ascending=False, inplace=True)
    ftrs.set_index('Feature', inplace=True)
    ftrs.to_csv(os.path.join(output_dir, "rf_features.csv"))
