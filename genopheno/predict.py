import argparse
import os
import pickle
import pandas as pd
from preprocessing.users import UserPhenotypes, User
from preprocessing.snp import extract_rsid, format_snps
from models.common import build_model_desc
from patsy import dmatrix
from util import setup_logger, timed_invoke, expand_path, clean_output

import logging.config
logger = logging.getLogger('root')


def run(users_dir, init_dir, model_dir, output_dir):
    """
    Predicts phenotype for users
    :param users_dir: The directory containing the user
    :param init_dir: The directory containing the preprocessed files
    :param model_dir: The directory containing the model files
    :param output_dir: The directory to write the predictions to
    """
    users_dir = expand_path(users_dir)
    init_dir = expand_path(init_dir)
    model_dir = expand_path(model_dir)
    output_dir = expand_path(output_dir)

    # Make sure output directory exists before doing work
    clean_output(output_dir)

    # Setup console and file loggers
    setup_logger(output_dir, "predict")

    # Read SNP data
    snp_details = pd.read_csv(os.path.join(init_dir, 'snp_database.csv.gz'), compression='gzip')

    # Read model config
    with open(os.path.join(model_dir, 'model_config.pkl'), 'rb') as f:
        model_config = pickle.load(f)

    # Filter snps to only include selected snps
    snp_columns = model_config['snps']
    selected_rsids = map(extract_rsid, snp_columns)
    snp_details = snp_details[snp_details['Rsid'].isin(selected_rsids)]


    # Predict for each user
    imputer = model_config['imputer']
    model_desc = build_model_desc(snp_columns, model_config['no_interactions'])
    model = model_config['model']
    pheno_map = model_config['pheno_map']
    users = []
    predictions = []

    def calc_mutations(user):
        mutations = user.allele_transformation(snp_details, how='right')
        mutations['Rsid'] = mutations['Rsid'].apply(format_snps, args=(snp_details,))
        mutations.set_index('Rsid', inplace=True)
        mutations = mutations.transpose()

        return mutations

    def predict_pheno(mutations):
        # Impute missing values
        x = imputer.transform(mutations)

        # Create model feature set
        x = dmatrix(model_desc, pd.DataFrame(x, columns=mutations.columns))

        # Predict
        return model.predict(x)[0]

    count = 0
    user_files = UserPhenotypes.get_user_geno_files(users_dir)
    for user_file in user_files:
        user = User(users_dir, user_file)
        count += 1

        # Calculate mutations
        mutations = timed_invoke('calculating mutations for user {} ({}/{})'.format(user.id, count, len(user_files)),
                                 lambda: calc_mutations(user))

        # Predict phenotype
        pheno_id = timed_invoke('prediction mutations for user {} ({}/{})'.format(user.id, count, len(user_files)),
                                lambda: predict_pheno(mutations))
        users.append(user.id)
        predictions.append(pheno_map[pheno_id])

    pd.DataFrame({'user_id': users, 'prediction': predictions})\
        .to_csv(os.path.join(output_dir, 'predictions.csv'), index=False, columns=['user_id', 'prediction'])

    print('Output written to "{}"'.format(output_dir))


if __name__ == '__main__':
    # Parse input
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--users-dir",
        "-u",
        metavar="<directory path>",
        default="resources" + os.sep + "data" + os.sep + "users",
        help="The directory that contains the users genomic data to predict the phenotypes for."
             "\n\nDefault: resources/data/users"
    )

    parser.add_argument(
        "--init-dir",
        "-i",
        metavar="<directory path>",
        default="resources" + os.sep + "full_data" + os.sep + "preprocessed",
        help="The directory that the preprocessed files are in."
             "\n\nDefault: resources/full_data/preprocessed"
    )

    parser.add_argument(
        "--model-dir",
        "-m",
        metavar="<directory path>",
        default="resources" + os.sep + "data" + os.sep + "model",
        help="The directory that the model files are in."
             "\n\nDefault: resources/data/model"
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="<directory path>",
        default="resources" + os.sep + "data" + os.sep + "prediction",
        help="The directory that the output files should be written to."
             "\n\nDefault: resources/data/prediction"
    )

    args = parser.parse_args()
    run(args.users_dir, args.init_dir, args.model_dir, args.output)
