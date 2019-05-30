
import numpy as np
import pandas as pd
import math

import logging
logger = logging.getLogger("root")


"""
The different possible mutation levels.
nm = no mutations
pm = partial mutations
fm = full mutations
"""
MUTATION_LEVELS = ['nm', 'pm', 'fm']


def __remove_missing_data(pheno, snp_data, invalid_thresh):
    """
    Removes missing data from the user data. If a SNP row has a percentage of users with an invalid or missing genotype
    then the SNP row is removed. Missing SNP data is changed from its numeric missing value to NaN.
    :param snp_data: The SNP data for all users
    :param invalid_thresh: The maximum percentage of invalid data for a row or column
    :return: The SNP data for all users with the missing data removed
    """
    non_user_columns = ['Gene_info', 'pct_fm', 'pct_nm', 'pct_pm']
    user_columns = set(snp_data.columns.values).difference(non_user_columns)
    users_count = len(user_columns)

    snp_count = snp_data.shape[0]

    min_required = math.ceil((1 - invalid_thresh / float(100)) * users_count)
    snp_data.dropna(axis=0, thresh=min_required, inplace=True, subset=user_columns)
    logger.info("{} ({:.2f}%) SNPs removed due to too many missing user observations for phenotype '{}'"
                .format(snp_count - snp_data.shape[0], float(snp_count - snp_data.shape[0]) / snp_count * 100, pheno))


def __filter_snps(row, abs_diff_thresh, relative_diff_thresh, selected_snps):
    """
    Determines if there is a significant mutation difference at each SNP between the two phenotype options.
    Each SNP contains the percentage of users with no, partial and full mutations. Each mutation level is compared
    respectively. If the mutation difference is greater than the defined difference and magnitude thresholds at any
    level, then the SNP is selected.
    :param row: The DataFrame row containing the mutation percentages. The index is the RSID.
    :param abs_diff_thresh: The difference threshold required for the SNP to be selected, as a percentage of the lower value.
    :param relative_diff_thresh:  The magnitude of the difference threshold, in percentage points. This is the
                              minimum value for the difference in mutation percentage divided by the minimum
                              mutation value out of the two phenotypes. The purpose of this is to filter out
                              SNPs where the change meets the difference threshold, but is still a small
                              magnitude. For example, if the mutation difference is 5%, but the SNP mutation
                              levels for each phenotype are 100% and 95% then this is less meaningful than if
                              the mutation levels were 6% and 1%. The magnitude threshold is meant to filter
                              out the SNP where the mutations are 100% and 95% and keep the SNP where the
                              mutations are 6% and 1%.
    :param selected_snps: A list containing the selected SNP RSIDs. If the row has a significant mutation difference,
                          then the RSID for this row will be added to the list.
    """
    # TODO: modify relative diff thresh description
    snp_rsid = row.name

    for mutation_level in MUTATION_LEVELS:
        # Get the mutation percentages at the mutation level for each phenotype
        mutation_pct_a = np.round(row['pct_{}_a'.format(mutation_level)], 3)
        mutation_pct_b = np.round(row['pct_{}_b'.format(mutation_level)], 3)

        # Calculate the mutation difference between phenotypes
        pct_diff = abs(mutation_pct_a - mutation_pct_b)

        # Check if the difference is greater than the defined difference threshold
        if pct_diff >= abs_diff_thresh:
            pct_min = min(mutation_pct_a, mutation_pct_b)
            if pct_min != 0:
                # Check if the difference is greater than the defined magnitude threshold
                if 100 * pct_diff / pct_min >= relative_diff_thresh:
                    selected_snps.append(snp_rsid)
                    break
            else:
                selected_snps.append(snp_rsid)
                break


def __select_snps(snp_pheno_pcts, m=-1.14, b=111):
    """
    Selects SNPs that have a significant difference in mutation percentage between phenotype groups based on a
    linear threshold that is modeled after the dominant-recessive disease model. For each SNP, `snp_pheno_pcts` has a
    record of the percent of users that have full, partial, or no mutation in each phenotype group.
    The equation for the linear threshold is:
        linear_thresh = -1.14 * max + 111
    where max is the higher mutation. SNPs are selected such the lower mutation percent value is less than or equal
    to the linear threshold.
    :param snp_pheno_pcts: The DataFrame row containing the mutation percentages. The index is the RSID.
    :param selected_snps: A list containing the selected SNP RSIDs. If the row has a significant mutation difference,
                          then the RSID for this row will be added to the list.
    :param m: slope for linear threshold (default: -1.14)
    :param b: intercept for linear threshold (default: 111)
    :return:
    """

    selected_snps = set()

    for mutation_level in MUTATION_LEVELS:
        thresh_df = pd.DataFrame()
        # extract mutation columns
        mutation_pct_a = snp_pheno_pcts['pct_{}_a'.format(mutation_level)].round(3)
        mutation_pct_b = snp_pheno_pcts['pct_{}_b'.format(mutation_level)].round(3)

        # # linear_thresh calculated using MIN
        # logger.info("linear thresh calculated using MIN")
        # logger.info("thresh = -1.05 * max + 105")
        # thresh_df["min"] = pd.concat([mutation_pct_a, mutation_pct_b], axis=1).min(axis=1)
        # # thresh_df["min"] = thresh_df["min"].apply(lambda min_val: 5 if min_val < 5 else min_val)
        # thresh_df["abs_diff"] = np.abs(mutation_pct_a - mutation_pct_b)
        # thresh_df["relative_diff"] = (thresh_df["abs_diff"] / thresh_df["min"]) * 100
        # # thresh_df["linear_thresh"] = m * thresh_df["min"] + b
        # thresh_df["linear_thresh"] = -1.05 * thresh_df["abs_diff"] + 105
        # thresh_df["linear_thresh"] = thresh_df["linear_thresh"].apply(lambda val: 10 if val < 10 else val)
        # selected_ids = thresh_df[(5 <= thresh_df["abs_diff"]) & (thresh_df["abs_diff"] <= 80) &
        #                          (thresh_df["relative_diff"] > thresh_df["linear_thresh"])].index

        # linear_thresh calculated using MAX
        thresh_df["min"] = pd.concat([mutation_pct_a, mutation_pct_b], axis=1).min(axis=1)
        thresh_df["max"] = pd.concat([mutation_pct_a, mutation_pct_b], axis=1).max(axis=1)
        # filter out those with max < 5
        thresh_df.drop(thresh_df[thresh_df["max"] <= 5].index, axis=0, inplace=True)
        # MAY 7 TEST: m = -1.07, b=105.35
        # m = -1.07
        # b = 105.35
        thresh_df["relative_diff"] = m * thresh_df["max"] + b
        thresh_df["relative_diff"] = thresh_df["relative_diff"].apply(lambda thresh: 20 if thresh < 20 else thresh)
        thresh_df["lower_thresh"] = (1 - (thresh_df["relative_diff"] / 100)) * thresh_df["max"]
        # filter snps based on lower <= low_thresh
        # thresh_df is indexed by SNP rsid
        selected_ids = thresh_df[(thresh_df["min"] <= thresh_df["lower_thresh"])].index

        # append selected snps to selected_snps array
        selected_snps.update(selected_ids)

    return list(selected_snps)


def __identify_mutated_snps(phenotypes, relative_diff_thresh):
    """
    Identifies SNPs of interest based on mutation differences between the two phenotypes.
    :param phenotypes: A dictionary of phenotypes where the key is the phenotype label and the value is the
                       DataFrame with the mutation information for all users with the phenotype.
    :param relative_diff_thresh: The relative difference in mutation percentage, calculated as a percent of the
                                larger mutation percent value.
    :return: A list of selected SNP RSIDs.
    """
    if len(phenotypes) != 2:
        raise ValueError('Using mutation differences to identify SNPs is only valid for two phenotype options.')

    # Merge the data frames top identify common SNPs and be able to do selection in one pass
    pheno_df_a = list(phenotypes.values())[0]
    pheno_df_b = list(phenotypes.values())[1]
    mutation_columns = ['pct_nm', 'pct_pm', 'pct_fm']
    merged = pheno_df_a[mutation_columns].merge(pheno_df_b[mutation_columns], left_index=True, right_index=True,
                                                suffixes=('_a', '_b'))

    if relative_diff_thresh:
        logger.info("using user defined threshold: {}".format(relative_diff_thresh))
        selected_snps = __select_snps(merged, m=0, b=relative_diff_thresh)
    else:
        selected_snps = __select_snps(merged)

    return selected_snps


def __format_selected_snps(pheno_label, pheno_df, selected_snps):
    """
    Builds the phenotype DataFrame used for the machine learning model based on the selected SNPs
    :param pheno_df: The DataFrame for the phenotype with the user mutation data
    :param selected_snps: A list of selected SNP RSIDs
    :param pheno_label: The phenotype label (i.e. 'Brown' for eye color)
    :return: The DataFrame for the selected SNPs
    """
    # Filter out SNPs that have not been selected
    snp_data = pheno_df.loc[selected_snps]
    snp_data.index = 'gene_' + snp_data['Gene_info'].str.replace(r'\W', '_') + '_' + snp_data.index

    # Drop the mutation percentages and gene info columns because they are no longer needed
    snp_data.drop(labels=['Gene_info', 'pct_fm', 'pct_nm', 'pct_pm'], axis=1, inplace=True)

    # Transpose the data and add columns for user Id and phenotype
    transposed_data = snp_data.transpose()
    transposed_data['phenotype'] = pheno_label

    return transposed_data


def create_dataset(phenotypes, invalid_thresh, invalid_user_thresh, relative_diff_thresh):
    """
    Function to return those SNPs that satisfy a criterion to check for differences between blue and brown SNPs
    :param phenotypes: A map of phenotypes where the key is the phenotype ID and the value is the phenotype data frame.
    :param invalid_thresh: The percentage of missing user observations a SNP can have before it is removed
    :param invalid_user_thresh: The acceptable percentage of missing data before a user is discarded
    :param relative_diff_thresh: The relative difference in mutation percent, calculated as a percent of the
                                larger mutation percent value.
    :return: A DataFrame where each row is a user and each column is a SNP.
    The value is the number of mutations (0,1,2).
    """
    # Filter out SNPs that do not have enough user observations
    for pheno, pheno_df in phenotypes.items():
        __remove_missing_data(pheno, pheno_df, invalid_thresh)

    # Select snps based on mutation differences between phenotypes
    selected_snps = __identify_mutated_snps(phenotypes, relative_diff_thresh)
    logger.info('{} SNPs with mutation differences identified'.format(len(selected_snps)))

    # Generate data frame for each phenotype using the selected SNPs
    final_datasets = []
    for pheno_key, pheno_df in phenotypes.items():
        final_datasets.append(__format_selected_snps(pheno_key, pheno_df, selected_snps))

    # Merge and return aggregate data set
    merged = pd.concat(final_datasets)

    # Remove users that do not have enough observations
    user_count = merged.shape[0]
    snp_count = merged.shape[1] - 1
    min_obs = math.ceil((1 - invalid_user_thresh / float(100)) * snp_count)
    merged.dropna(axis=0, thresh=min_obs, inplace=True)
    logger.info('{} users dropped due to too many missing observations'.format(user_count - merged.shape[0]))
    logger.info("Model Data contains {} users and {} SNPs".format(merged.shape[0], snp_count))

    return merged
