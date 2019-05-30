import os
import re

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger('root')

MUTATION_LEVELS = ['nm', 'pm', 'fm']


class UserPhenotypes:
    """
    Represents a collection of users with known phenotypes
    """

    def __init__(self, known_pheno_file, user_data_dir):
        """
        Creates a new UserPhenos object
        :param known_pheno_file: The file path containing the known use phenotype classifications
        :param user_data_dir: A dictionary where the key is the phenotype classification value and the
        value is a list of users
        """
        self.__phenotypes = self.__map_phenotypes(known_pheno_file, user_data_dir)

    def __map_phenotypes(self, known_pheno_file, user_data_dir):
        """
        Maps each user data file to a phenotype classification
        :param known_pheno_file: The file path containing the known use phenotype classifications
        :param user_data_dir: The directory path containing all user data files
        :return: A dictionary where the key is the phenotype classification value and the value is a list of users
        with the phenotype.
        """
        phenotype_classifications = pd.read_csv(known_pheno_file)
        phenotypes_map = {}
        no_pheno = []
        users = []
        duplicates = []
        multi_pheno = []

        for user_file_name in self.get_user_geno_files(user_data_dir):
            # OpenSNP sometimes contains two genomic files for the same user Id. This is used to avoid duplicate
            # information for a user. The first file for the user is used.
            user = User(user_data_dir, user_file_name)
            if user.id in users:
                logger.warning('User {} already is associated with a genomic file. Ignoring file "{}"' \
                               .format(user.id, user.file_path))
                duplicates.append(user.id)
                continue

            # Get the phenotype classification for the user
            phenotype_row = \
                phenotype_classifications.loc[phenotype_classifications['user_id'] == user.id]['phenotype'].values
            if phenotype_row.size < 1:
                no_pheno.append(user.id)
                continue
            elif phenotype_row.size > 1:
                # handle duplicate entries that all have the same value
                phenotype_row = np.unique(phenotype_row)
                if len(phenotype_row) > 1:
                    logger.warning('Found {} phenotype classifications for user {}. Each user should '
                                   'have a single classification.'.format(phenotype_row.size, user.id))
                    multi_pheno.append(user.id)
                    continue

            phenotype = phenotype_row[0]

            # map the user to the phenotype
            phenotype_group = phenotypes_map.get(phenotype)
            if phenotype_group is None:
                phenotype_group = []
                phenotypes_map[phenotype] = phenotype_group

            phenotype_group.append(user)
            users.append(user.id)

        if len(no_pheno) > 0:
            logger.warning('No phenotype classification for {} users {}.'.format(len(no_pheno), no_pheno))
        if len(duplicates) > 0:
            logger.warning('Multiple genotype files found for {} users: {}.'
                           .format(len(duplicates), duplicates))
        if len(multi_pheno) > 0:
            logger.warning('Multiple phenotypes found for {} users: {}.'
                           .format(len(multi_pheno), multi_pheno))

        return phenotypes_map

    @staticmethod
    def get_user_geno_files(user_data_dir):
        """
        Gets the list of user files. Each file contains the genetic information of one user. 23andMe and Ancestry.com
        data formats are supported.

        File names must start with the user id followed by an underscore and end with either 23andme.txt or ancestry.txt.
        Examples:
        user44_file19_yearofbirth_1970_sex_XY.23andme.txt
        user44_file19_yearofbirth_1970_sex_XY.ancestry.txt
        :param user_data_dir: The directory containing the user files.
        :return: The list of user files in the directory.
        """
        file_name_regex = re.compile("^user[0-9]+_.*(23andme|ancestry).txt$")
        files = os.listdir(user_data_dir)
        return list(filter(file_name_regex.match, files))

    def reduce_phenotypes(self, reducer):
        """
        Invokes a method for each known user phenotype
        :param reducer: The phenotype function. This should accept a phenotype key and a list of users
        :return:
        """
        for phenotype, users in self.__phenotypes.items():
            self.__phenotypes[phenotype] = reducer(phenotype, users)


class User:
    """
    Represents a user's genetic information
    """

    def __init__(self, user_data_dir, user_file_name):
        """
        Creates a user
        :param user_data_dir: The directory where the user data file is located
        :param user_file_name: The user data file name
        """
        self.file_path = os.path.join(user_data_dir, user_file_name)
        self.__set_id(user_file_name)

    def __set_id(self, user_file_name):
        # Extract just the user id from the file name
        # The file must start with user<numeric_id>
        match = re.search('^user[0-9]*', user_file_name)
        user_id = match.group(0)
        # remove user prefix
        user_id = int(user_id[len('user')::])
        self.id = user_id

    def __get_data_frame(self, snp_details, how):
        """
        Gets the user data and joins it with the SNP data. The user data only includes the SNP accession number and not
        the details. This creates the user data frame and adds the SNP details to it.
        :param snp_details: The gene_data is used to add the SNP data (i.e. ref and alt) to the user data.
        The user data includes the accession number (rsid), but not the details for the SNP.
        :param how: The data frame merge method (i.e. inner)
        :return: The user data, joined with the SNP data, as a data frame. The data frame has columns Rsid, Genotyoe,
        Ref, Alt, Gene_info.
        """
        try:
            if self.file_path.endswith("23andme.txt"):
                data_person = pd.read_table(self.file_path, header=None, comment="#", low_memory=False,
                                            error_bad_lines=False, delim_whitespace=True, warn_bad_lines=False)
                data_person.columns = ["Rsid", "chromosome", "position", "genotype"]
                data_person.drop(["chromosome", "position"], axis=1, inplace=True)
            elif self.file_path.endswith("ancestry.txt"):
                data_person = pd.read_table(self.file_path, header=0, comment="#", low_memory=False,
                                            error_bad_lines=False, delim_whitespace=True, warn_bad_lines=False)
                data_person.columns = ["Rsid", "chromosome", "position", "allele1", "allele2"]
                data_person["genotype"] = data_person["allele1"] + data_person["allele2"]
                data_person = data_person.loc[data_person["Rsid"].isin(list(snp_details["Rsid"]))]
                data_person.drop(["chromosome", "position", "allele2", "allele1"], axis=1, inplace=True)
            else:
                raise ValueError('Only 23andMe and Ancestry.com data formats are supported')
        except Exception as e:
            logger.warning('{} does not contain valid user genomic data. Skipping user. ' \
                           'Reason: {}'.format(self.file_path, e))
            return pd.DataFrame()

        data_person.columns = ['Rsid', 'Genotype']

        # drop public RSIDs. It was found that some genomic files from OpenSNP have duplicate RSID values
        duplicates = data_person[data_person.duplicated(subset=['Rsid'])].values
        if len(duplicates) > 0:
            logger.warning('User {} has duplicate RSID values. The duplicates will be removed.{}{}'\
                           .format(os.linesep, self.id, duplicates))
            data_person.drop_duplicates(subset=['Rsid'], inplace=True)

        data = pd.merge(data_person, snp_details[['Rsid', 'Ref', 'Alt']], how=how, on=["Rsid"], right_index=True)

        return data

    def allele_transformation(self, snp_details, how='inner'):
        """
        Gets the user genetic data and counts the number of mutations for each gene.
        :param snp_details: The data frame containing the SNP data
        :param how: The data frame merge method (i.e. inner)
        :return: The data frame containing the users genetic data. The data frame has columns Rsid, Gene_info and
        <user_id>_snp_<phenotype> where <user_id>_snp_<phenotype> is the number of mutations the user has for the gene.
        """
        data = self.__get_data_frame(snp_details, how)
        if not data.empty:
            # count the number of mutations for each user SNP
            data[self.id] = data.apply(self.__count_mutations, axis=1)

            # once mutations are counted the genotype, ref and alt columns are no longer needed
            data.drop(['Genotype', 'Ref', 'Alt'], axis=1, inplace=True)

        return data

    def __count_mutations(self, row):
        """
        Counts the number of mutations the the user has for each SNP.
        :param row: A data frame row with columns Genotype, Ref and Alt
        :return: The number of mutations the user has for the SNP
        """
        try:
            genotype = list(row['Genotype'])
            ref = row['Ref']
            alt = list(row['Alt'])

            if len(genotype) != 2:
                return np.nan

            for nucleotide in genotype:
                if nucleotide != ref and nucleotide not in alt:
                    return np.nan

            changes = 0
            for nucleotide in genotype:
                if nucleotide != ref:
                    changes += 1

            return changes
        except Exception as e:
            logger.warning('Invalid SNP for user {}. Marking invalid.{}Row: {}.' \
                           '{}Reason: {}'.format(self.id, os.linesep, row, os.linesep, e))
            return np.nan
