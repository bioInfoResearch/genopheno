

import sklearn.metrics as skm
import matplotlib as mp
mp.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from patsy import ModelDesc, EvalFactor, Term, dmatrix
__all__ = [ModelDesc, EvalFactor, Term, dmatrix]
from os import linesep, path
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc

import logging
logger = logging.getLogger("root")


def build_model(data_set, data_split, no_interactions, negative, model, cross_validation, max_snps, output_dir,
                param_grid={}, model_eval={}):
    """
    Builds a model for the data set
    :param data_set: The data set (training and testing)
    :param data_split: The percentage of data that should be used for testing
    :param no_interactions: If false interactions aren't included in the model
    :param negative: The negative phenotype label
    :param model: The model to use for training and testing the data
    :param cross_validation: The number of folds for k-fold cross validation
    :param max_snps: The maximum number of SNPs for the model to include
    :param output_dir: The directory to write the model artifacts in
    :param param_grid: The parameter matrix for the model
    :param model_eval: A dictionary of optional model evaluation methods
    """
    model_config = {}

    # Split the data into testing and training data
    x = data_set.drop(labels=['phenotype'], axis=1)
    snp_columns = x.columns.values
    if (max_snps is not None) and (len(snp_columns) > max_snps):
        logger.warning('Too many model SNPs ({}, configured max: {}). Dropping extra SNPs.'
                       .format(len(snp_columns), max_snps))
        snp_columns = snp_columns[:max_snps]
        x = x[snp_columns]
    model_config['snps'] = snp_columns
    y = data_set['phenotype']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=data_split/float(100), random_state=1, stratify=y
    )

    # Convert classifications to 0 and 1
    #
    # Do this after the test train split because while the ratio of each phenotype is the same in the split, different
    # rows are chosen based on which phenotype is assigned a 0 and 1. Do this after the split means consistent rows
    # will be selected regardless of which phenotype is assigned as the negative.
    pheno_map = __pheno_to_binary(y_train, y_test, negative)
    model_config['pheno_map'] = pheno_map

    # Replace nan values
    imputer, x_train, x_test = __impute_data(x_train, x_test)
    model_config['imputer'] = imputer

    # print data counts
    __save_data_summary(pheno_map, y_train, y_test, len(snp_columns), output_dir)

    # Define model
    model_config['no_interactions'] = no_interactions
    model_desc = build_model_desc(snp_columns, no_interactions)
    x_train = dmatrix(model_desc, pd.DataFrame(x_train, columns=snp_columns))
    x_test = dmatrix(model_desc, pd.DataFrame(x_test, columns=snp_columns))

    # Fit training data to model
    grid = GridSearchCV(model, param_grid=param_grid, cv=cross_validation, verbose=5)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    model_config['model'] = best_model
    __save_model(model_config, output_dir)
    logger.info('Best estimator params found during grid search: {}'.format(grid.best_params_))

    # Test model
    y_pred = best_model.predict(x_test)
    __save_confusion_matrix(y_test, y_pred, output_dir, 'testing_data')
    __save_confusion_matrix(y_train, best_model.predict(x_train), output_dir, 'training_data')

    # Optional model stats
    roc_probs = model_eval.get('roc')
    if roc_probs:
        __save_roc(y_test, roc_probs(best_model, x_test), output_dir)

    features = model_eval.get('features')
    if features:
        model_terms = __get_model_term_labels(model_desc)
        features(best_model, model_terms, output_dir)


def __save_confusion_matrix(y_true, y_pred, output_dir, file_suffix):
    """
    Calculates the metrics for the model prediction using a confusion matrix
    :param y_true: The test data provided as a numpy array
    :param y_pred: The test predicted by the model as a numpy array
    :param output_dir: The directory to write the results to
    :param file_suffix: The suffix for the output file name
    """
    confusion_matrix = skm.confusion_matrix(y_true, y_pred)
    true_pos = confusion_matrix[1][1]
    true_neg = confusion_matrix[0][0]
    false_pos = confusion_matrix[0][1]
    false_neg = confusion_matrix[1][0]

    accuracy = float(true_pos + true_neg)/float(true_pos + true_neg + false_neg + false_pos)
    sensitivity = float(true_pos)/float(true_pos + false_neg)
    specificity = float(true_neg)/float(true_neg + false_pos)

    metrics = 'Confusion Matrix Metrics: {}    Accuracy:    {}{}    Sensitivity: {}{}    Specificity: {}{}    ' \
              'TP: {}{}    TN: {}{}    FP: {}{}    FN: {}{}'\
        .format(linesep, np.round(accuracy, 3),
                linesep, np.round(sensitivity, 3),
                linesep, np.round(specificity, 3),
                linesep, np.round(true_pos, 3),
                linesep, np.round(true_neg, 3),
                linesep, np.round(false_pos, 3),
                linesep, np.round(false_neg, 3),
                linesep)

    logger.info(metrics)
    with open(path.join(output_dir, 'confusion_matrix_{}.txt'.format(file_suffix)), 'w') as metrics_file:
        metrics_file.write(metrics)


def __pheno_to_binary(y_train, y_test, negative):
    """
    Converts the phenotype labels to 0 and 1
    :param data_set: The feature matrix
    :param negative: The phenotype label that should be negative
    :returns The phenotype numeric to string label mapping
    """
    # Identify negative phenotype
    phenotypes = set(y_train)
    phenotypes.update(y_test)
    phenotypes = sorted(phenotypes)
    if negative is None:
        negative = phenotypes[0]
    elif negative not in phenotypes:
        raise ValueError('{} is an invalid negative phenotype option. Must be one of the following: {}'
                         .format(negative, phenotypes))

    # Identify positive phenotype
    phenotypes.remove(negative)
    positive = phenotypes[0]

    # Replace the string values with binary values
    y_train.replace([negative, positive], [0, 1], inplace=True)
    y_test.replace([negative, positive], [0, 1], inplace=True)

    return {0: negative, 1: positive}


def __impute_data(x_train, x_test):
    """
    Fills in the missing data. nan values will be replaced with the most frequent value for the feature
    :param x_train: The training data
    :param x_test: The test data
    :return: The fitted imputer, modified training and test data.
    """
    imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, copy=True, verbose=1)

    train_snp_count = x_train.shape[1]
    x_train = imputer.fit_transform(x_train)
    if train_snp_count > x_train.shape[1]:
        raise ValueError('A SNP column was dropped while imputing the training set. '
                         'This means the entire feature had no data. Try decreasing the invalid SNP threshold.')

    test_snp_count = x_test.shape[1]
    x_test = imputer.transform(x_test)
    if test_snp_count > x_test.shape[1]:
        raise ValueError('A SNP column was dropped while imputing the test set. '
                         'This means the entire feature had no data. Try decreasing the invalid SNP threshold.')

    return imputer, x_train, x_test


def build_model_desc(snps, no_interactions):
    """
    Creates the model description (formula)
    :param snps: The selected snp labels
    :param no_interactions: If false, interactions will not be included in the model
    :return: The model description
    """
    x_terms = []
    for i in range(len(snps)):
        # Main effects
        snp_i = EvalFactor(snps[i])
        x_terms.append(Term([snp_i]))

        if not no_interactions:
            for j in range(i + 1, len(snps)):
                # Interaction effects
                snp_j = EvalFactor(snps[j])
                x_terms.append(Term([snp_i, snp_j]))

    return ModelDesc([], x_terms)


def __get_model_term_labels(model_desc):
    term_labels = []
    for term in model_desc.rhs_termlist:
        term_label = ':'.join(exp.code for exp in term.factors)
        term_labels.append(term_label)

    return term_labels


def __save_feature_importance(model, model_desc, output_dir):
    """
    Saves the most influential model features, sorted. Since all features are scaled to the same range
    the coefficients can be used to evaluate feature importance.
    :param model: The fitted model
    :param model_desc: The model description
    :param output_dir: The directory to write the feature importance in
    """
    if hasattr(model, 'coef_'):
        term_labels = []
        for term in model_desc.rhs_termlist:
            term_label = ':'.join(exp.code for exp in term.factors)
            term_labels.append(term_label)

        features = pd.DataFrame({'feature': term_labels, 'coefficient': model.coef_.ravel()})
        features['coef_abs'] = features['coefficient'].abs()
        features = features[features['coef_abs'] > 0]
        features.sort_values(ascending=False, inplace=True, by='coef_abs')

        with open(path.join(output_dir, 'features.csv'), 'w') as f:
            f.write('intercept: {}{}{}'.format(model.intercept_, linesep, linesep))
            features[['feature', 'coefficient']].to_csv(f, index=False)


def __save_model(model_config, output_dir):
    """
    Persists the imputer and model so that it can used with new data
    :param model_config: The dictionary containing all model objects needed to make predictions on new data
    :param output_dir: The directory to write the imputer and model to
    """
    try:
        with open(path.join(output_dir, 'model_config.pkl'), 'wb') as f:
            f.write(pickle.dumps(model_config))
    except Exception as e:
        logger.info('Cannot save model: {}'.format(e))


def __save_roc(y_true, y_pred, output_dir):
    """
    Creates an ROC curve with AUC for the model
    :param y_true: The actual phenotypes for the test data
    :param y_pred: The predicted phenotypes for the test data
    :param output_dir: The directory to save the ROC curve in
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot code referenced from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(path.join(output_dir, 'roc.png'))
    plt.close()


def __save_data_summary(pheno_map, y_train, y_test, n_snps, output_dir):
    # counts for phenotypes
    n_neg = len(y_train[y_train == 0]) + len(y_test[y_test == 0])
    n_pos = len(y_train[y_train == 1]) + len(y_test[y_test == 1])
    total = n_neg + n_pos

    # percent of phenotypes in data
    p_neg = np.round(float(n_neg) / float(total) * 100, 1)
    p_pos = np.round(float(n_pos) / float(total) * 100, 1)

    # negative and positive labels
    neg = pheno_map[0]
    pos = pheno_map[1]

    # model data counts
    n_train = len(y_train)
    n_test = len(y_test)

    # format summary
    pheno_summary = "-- Data Summary --{}\tNegative ({}):\t{} ({}%){}\tPositive ({}):\t{} ({}%){}\tTOTAL:\t{}{}"\
        .format(linesep, neg, n_neg, p_neg,
                linesep, pos, n_pos, p_pos,
                linesep, total, linesep)
    count_summary = "Training Count:\t{}{}Test Count:\t{}{}Number of SNP Features:\t{}{}"\
        .format(n_train, linesep,
                n_test, linesep,
                n_snps, linesep)

    logger.info(pheno_summary)
    logger.info(count_summary)

    # write to file
    with open(path.join(output_dir, 'data_summary.txt'), 'w') as summary_file:
        summary_file.write(pheno_summary)
        summary_file.write(count_summary)
    summary_file.close()
