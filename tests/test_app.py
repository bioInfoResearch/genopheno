import argparse
import shutil
import gzip
import filecmp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from genopheno import preprocess, model, predict
from os import mkdir
from os.path import join, exists

EXPECTED_DIR = join('tests', 'resources', 'expected')
EXPECTED_PREPROCESS = join(EXPECTED_DIR, 'preprocess')
EXPECTED_MODEL = join(EXPECTED_DIR, 'model')
EXPECTED_PREDICT = join(EXPECTED_DIR, 'predict')
DATA_DIR = join('genopheno', 'resources', 'data')
USER_DATA_DIR = join(DATA_DIR, 'users')
SNP_DATA_DIR = join(DATA_DIR, 'snp')
KNOWN_PHENOTYPES_FILE = join(DATA_DIR, 'known_phenotypes.csv')
OUTPUT_DIR_PREPROCESS = 'output_preprocess'
OUTPUT_DIR_MODEL = 'output_model'
OUTPUT_DIR_PREDICT = 'output_predict'


def test_preprocess_eye_color():
    """
    Tests the preprocessing step.
    """
    __create_dir(OUTPUT_DIR_PREPROCESS)
    preprocess.run(USER_DATA_DIR, SNP_DATA_DIR, KNOWN_PHENOTYPES_FILE, OUTPUT_DIR_PREPROCESS)
    __assert_files(
            ['preprocessed_Blue_Green.csv.gz', 'preprocessed_Brown.csv.gz', 'snp_database.csv.gz'], EXPECTED_PREPROCESS, OUTPUT_DIR_PREPROCESS
        )


def test_model_eye_color(ml_model):
    """
    Tests the modeling step.
    """
    __create_dir(OUTPUT_DIR_MODEL)
    global EXPECTED_MODEL
    if ml_model == 'en':
        EXPECTED_MODEL += "/elasticNet"
        model.run(EXPECTED_PREPROCESS, 50, 80, 15, 33, False, None, 200, ml_model, 3, OUTPUT_DIR_MODEL)
        __assert_files(
            ['confusion_matrix_training_data.txt', 'confusion_matrix_testing_data.txt', 'model_features.csv', 'model_config.pkl', 'roc.png'], EXPECTED_MODEL, OUTPUT_DIR_MODEL
        )
    elif ml_model == 'dt':
        EXPECTED_MODEL += "/decisionTree"
        model.run(EXPECTED_PREPROCESS, 50, 80, 15, 33, False, None, 200, ml_model, 3, OUTPUT_DIR_MODEL)
        __assert_files(
            ['confusion_matrix_training_data.txt', 'confusion_matrix_testing_data.txt', 'dtree.dot', 'dtree.png', 'model_config.pkl'], EXPECTED_MODEL, OUTPUT_DIR_MODEL
        )
    elif ml_model == 'rf':
        EXPECTED_MODEL += "/randomForest"
        model.run(EXPECTED_PREPROCESS, 50, 80, 15, 33, False, None, 200, ml_model, 3, OUTPUT_DIR_MODEL)
        __assert_files(
            ['confusion_matrix_training_data.txt', 'confusion_matrix_testing_data.txt', 'rf_features.csv', 'model_config.pkl', 'rf_features.csv'], EXPECTED_MODEL, OUTPUT_DIR_MODEL
        )
    elif ml_model == 'xg':
        EXPECTED_MODEL += "/xgBoost"
        model.run(EXPECTED_PREPROCESS, 50, 80, 15, 33, False, None, 200, ml_model, 3, OUTPUT_DIR_MODEL)
        __assert_files(
            ['confusion_matrix_training_data.txt', 'confusion_matrix_testing_data.txt', 'xg_features.csv', 'model_config.pkl', 'roc.png'], EXPECTED_MODEL, OUTPUT_DIR_MODEL
        )


def test_predict_eye_color():
    """
    Tests the prediction step.
    """
    __create_dir(OUTPUT_DIR_PREDICT)
    predict.run(USER_DATA_DIR, EXPECTED_PREPROCESS, EXPECTED_MODEL, OUTPUT_DIR_PREDICT)
    __assert_files(['predictions.csv'], EXPECTED_PREDICT, OUTPUT_DIR_PREDICT)


def __assert_files(files, exp_dir, act_dir):
    """
    Verifies that the files in both directories are the same
    :param files: An array of file names to compare
    :param exp_dir: The directory with the expected files
    :param act_dir: The directory with the actual files generated in the test
    """
    for filename in files:
        assert os.path.isfile(join(exp_dir, filename)) and os.path.isfile(join(act_dir, filename))


def __create_dir(directory):
    """
    Creates a directory. If present, the old directory and all files within the directory will be deleted.
    :param directory: The directory to create.
    """
    if exists(directory):
        shutil.rmtree(directory)
    mkdir(directory)

def run(model):
    test_preprocess_eye_color()
    test_model_eye_color(model)
    test_predict_eye_color()
    print("YOU HAVE SUCCESFULLY PASSED ALL TESTS!")

if __name__ == '__main__':

    # Parse input
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--model",
        "-m",
        default="en",
        type=str,
        help="The type of model to use."
             "\nen = Elastic net"
             "\ndt = Decision tree"
             "\nrf = Random Forest"
             "\nxg = XGboost"
             "\n\n Default: en"
    )

    args = parser.parse_args()
    run(args.model)

