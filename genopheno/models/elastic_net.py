from models import common
import pandas as pd
from os import path, linesep
from sklearn.linear_model import SGDClassifier


def build_model(data_set, data_split, no_interactions, negative, max_snps, cross_validation, output_dir):
    """
    Builds a model using logistic regression and an elastic net penalty
    :param data_set: The feature data set
    :param data_split: The percentage of data to use for testing the model
    :param no_interactions: If True interactions will not be included in the model
    :param negative: The negative phenotype label
    :param max_snps: The maximum number of SNPs for the model to include
    :param output_dir: The directory to write the model to
    """
    l1_ratio = 0
    l1_ratios = []
    step_size = 0.05
    while l1_ratio < 1:
        l1_ratios.append(l1_ratio)
        l1_ratio += step_size

    param_grid = {'l1_ratio': l1_ratios}

    model_eval = {
        'roc': get_roc_probs,
        'features': save_features
    }

    common.build_model(
        data_set,
        data_split,
        no_interactions,
        negative,
        SGDClassifier(
            loss="log", penalty="elasticnet", random_state=1, n_jobs=-1, max_iter=1000, tol=1e-3),
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
    return model.decision_function(x_test)


def save_features(model, term_labels, output_dir):
    """
    Saves the features ordered by influence. The reason the coefficients can be used to determine feature importance
    is because all feature data is scaled to be the same range before training the model.
    :param model: The trained model
    :param term_labels: The model term labels
    :param output_dir: The directory to write the features to
    """
    features = pd.DataFrame({'feature': term_labels, 'coefficient': model.coef_.ravel()})
    features['coef_abs'] = features['coefficient'].abs()
    features = features[features['coef_abs'] > 0]
    features.sort_values(ascending=False, inplace=True, by='coef_abs')
    features.drop('coef_abs', axis=1, inplace=True)

    with open(path.join(output_dir, 'model_features.csv'), 'w') as f:
        f.write('intercept: {}{}{}'.format(model.intercept_[0], linesep, linesep))

        # main effects
        f.write('{}main effects:{}'.format(linesep, linesep))
        main_effects = features['feature'].str.contains(':')
        features[~main_effects].to_csv(f, index=False)

        # interactions
        f.write('{}interaction effects:{}'.format(linesep, linesep))
        features[main_effects].to_csv(f, index=False)
