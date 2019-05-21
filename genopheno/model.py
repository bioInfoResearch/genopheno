import argparse
import os
import re

import pandas as pd
import logging
import logging.config

from models.snp_selectors import mutation_difference
from models import elastic_net, decision_tree, random_forest, xgboost_
from util import timed_invoke, expand_path, clean_output, setup_logger

logger = logging.getLogger('root')

MODELS = {
    'en': elastic_net.build_model,
    'dt': decision_tree.build_model,
    'rf': random_forest.build_model,
	'xg': xgboost_.build_model,
}


def __read_phenotype_input(input_dir):
    """
    Reads the preprocessed phenotype files from the initialization steps and creates data frames from each file.
    :param input_dir: The directory containing the preprocessed files.
    :return: A map of phenotypes where the key is the phenotype ID and the value is the phenotype data frame.
    """
    # read preprocessed files
    file_prefix = 'preprocessed_'
    file_name_regex = re.compile('^{}.+\.csv.gz$'.format(file_prefix))
    files = os.listdir(input_dir)

    phenotypes = {}
    for f in filter(file_name_regex.match, files):
        # create data frame from preprocessed files
        df = pd.read_csv(os.path.join(input_dir, f), compression='gzip')
        df.set_index('Rsid', inplace=True)

        # add the data frame to the collection of preprocessed phenotypes
        phenotype = f[len(file_prefix):len(f) - len('.csv.gz')]
        phenotypes[phenotype] = df
        logger.info("{} users and {} SNPs for phenotype '{}'".format(len(df.columns)-4, df.shape[0], phenotype))

    if len(phenotypes) == 0:
        raise ValueError('No preprocessed files in directory "{}". '
                         'This directory should contain the output from the preprocess step.'.format(input_dir))
    return phenotypes


def run(preprocessed_dir, invalid_thresh, invalid_user_thresh, relative_diff_thresh, data_split,
        no_interactions, negative, max_snps, model_id, cross_validation, output_dir):
    """
    Builds a model to predict phenotype
    :param preprocessed_dir: The directory containing the preprocessed data
    :param invalid_thresh: The acceptable percentage of missing data before a SNP is discarded
    :param invalid_user_thresh: The acceptable percentage of missing data before a user is discarded
    :param relative_diff_thresh: The relative difference in mutation percent, calculated as a percent of the
                                larger mutation percent value.
    :param data_split: The percent data used for testing.
    :param no_interactions: If True the model will not contain interactions
    :param negative: The negative phenotype label
    :param model_id: The id for the model to use
    :param cross_validation: number of folds for cross validation
    :param output_dir: The directory to write the model in
    """

    # Expand file paths
    preprocessed_dir = expand_path(preprocessed_dir)

    # Make sure output directory exists before doing work
    clean_output(output_dir)

    setup_logger(output_dir, model_id + "_model")

    # Get model
    build_model = MODELS.get(model_id)
    if not build_model:
        raise ValueError('Model Id "{}" is not valid'.format(model_id))

    phenotypes = timed_invoke('reading the preprocessed files', lambda: __read_phenotype_input(preprocessed_dir))

    data_set = timed_invoke('creating model data set', lambda: mutation_difference.create_dataset(
                               phenotypes, invalid_thresh, invalid_user_thresh, relative_diff_thresh)
                            )
    export_csv = data_set.to_csv(r'./exported.csv')



    timed_invoke('building model', lambda: build_model(data_set, data_split, no_interactions, negative, max_snps,
                                                       cross_validation, output_dir))
    logger.info('Output written to "{}"'.format(output_dir))


if __name__ == '__main__':

    # Parse input
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--preprocessed",
        "-p",
        metavar="<directory path>",
        default="resources" + os.sep + "full_data" + os.sep + "preprocessed",
        help="The directory containing the output data from the initialization phase."
             "\n\nDefault: resources/full_data/preprocessed"
    )

    parser.add_argument(
        "--invalid-snp-thresh",
        "-it",
        metavar="percent",
        type=float,
        default=60,
        help="The maximum percentage of missing or invalid user observations a SNP can have before it is not "
             "considered as a feature in the model."
             "\n\nDefault: 60"
    )

    parser.add_argument(
        "--invalid-user-thresh",
        "-iu",
        metavar="percent",
        type=float,
        default=90,
        help="The maximum percentage of missing or invalid SNP observations a user can have before it is not "
             "considered as a valid example in the model."
             "\n\nDefault: 90"
    )

    parser.add_argument(
        "--relative-diff-thresh",
        "-rdt",
        metavar="percent",
        type=float,
        help="The relative difference in percent of users with a particular genotype, used for SNP selection."
             "\n SNPs will be selected such that the lower percent value between the two phenotype groups is "
             "less than or equal to (1 - (relative_diff_thresh/100)) * higher. "
    )

    parser.add_argument(
        "--split",
        "-s",
        metavar="percent",
        type=float,
        default=33,
        help="The percentage of users to use as test data. The remaining users will be used for training the model."
             "\n\nDefault: 33"
    )

    parser.add_argument(
        "--no-interactions",
        "-ni",
        default=False,
        action='store_true',
        help="If set then interactions will not be included in the model. Note, this does not apply for the decision "
             "tree model."
             "\n\nDefault: False"
    )

    parser.add_argument(
        "--negative",
        "-n",
        help="The phenotype value that should be considered as the negative case. "
             "If not set, the alphabetically first phenotype value will be used as the negative value."
    )

    parser.add_argument(
        "--max-snps",
        "-ms",
        default=None,
        type=int,
        help="The maximum number of SNPs to include in the model"
             "\n\nDefault: None"
    )

    parser.add_argument(
        "--model",
        "-m",
        default="xg",
        type=str,
        help="The type of model to use."
             "\nen = Elastic net"
             "\ndt = Decision tree"
             "\nrf = Random Forest"
			 "\nxg = XGboost"
             "\n\n Default: xg"
    )

    parser.add_argument(
        "--cross-validation",
        "-cv",
        type=int,
        default=3,
        help="Number of folds for k-fold cross validation."
             "\n\nDefault: 3"
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="<directory path>",
        default="resources" + os.sep + "data" + os.sep + "model",
        help="The directory that the output files should be written to. This will include all files required for the "
             "machine learning input."
             "\n\nDefault: resources/data/model"
    )

    args = parser.parse_args()

    run(args.preprocessed, args.invalid_snp_thresh, args.invalid_user_thresh, args.relative_diff_thresh,
        args.split, args.no_interactions, args.negative, args.max_snps, args.model, args.cross_validation,
        args.output)
