"""
This is prototype code that is not used in the final implementation. This uses SNP dominant-recessive characteristics
to determine which SNPs will be used in the genotype to phenotype model. This was not chosen because it does not
account for factors that are not related to dominant and recessive characteristics alone, like eye color.
"""
import pandas as pd
import csv
import os


def dominant_check(upper, lower):
    data_bg = pd.read_csv('Data/output_temp/bg_snp_pct.csv')

    print("Dominant Blue Traits...")
    data_bg.sort_values(by=['pct_nm'], ascending=False, inplace=True)

    bg_nm_gtupper = data_bg.loc[data_bg['pct_nm'] >= upper]

    print("Number of SNPs having No mutations > %d: %d" % (upper, len(bg_nm_gtupper)))

    if len(bg_nm_gtupper) < int(0.01 * len(data_bg)):
        n_rows = int(len(data_bg) * 0.01)
        print(n_rows)
        bg_nm_gtupper = data_bg.head(n=n_rows)

    bg_nm_ltlower = data_bg.loc[data_bg['pct_nm'] <= lower]

    print("Number of SNPs having No mutations < %d: %d" % (lower, len(bg_nm_ltlower)))

    if len(bg_nm_ltlower) < int(0.01 * len(data_bg)):
        n_rows = int(len(data_bg) * 0.01)
        print(n_rows)
        bg_nm_ltlower = data_bg.tail(n=n_rows)

    # Brown

    print("Dominant Brown Traits...")
    data_br = pd.read_csv('Data/output_temp/br_snp_pct.csv')

    data_br.sort_values(by=['pct_nm'], ascending=False, inplace=True)

    br_nm_gtupper = data_br.loc[data_br['pct_nm'] >= upper]

    print("Number of SNPs having No mutations > %d: %d" % (upper, len(br_nm_gtupper)))

    if len(br_nm_gtupper) < int(0.01 * len(data_br)):
        n_rows = int(len(data_br) * 0.01)
        print(n_rows)
        br_nm_gtupper = data_br.head(n=n_rows)

    br_nm_ltlower = data_br.loc[data_br['pct_nm'] <= lower]

    print("Number of SNPs having No mutations < %d: %d" % (lower, len(br_nm_ltlower)))

    if len(br_nm_ltlower) < int(0.01 * len(data_br)):
        n_rows = int(len(data_br) * 0.01)
        print(n_rows)
        br_nm_ltlower = data_br.tail(n=n_rows)

    # ex:  brown_dominant as those Rsids with (bg_nm_pct <= 25 and br_nm_pct >= 75)
    brown_dominant = set(bg_nm_ltlower['Rsid']).intersection(br_nm_gtupper['Rsid'])
    blue_dominant = set(br_nm_ltlower['Rsid']).intersection(bg_nm_gtupper['Rsid'])

    csvfile = 'Data/output_temp/blue_dominant_snps.csv'

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in blue_dominant:
            writer.writerow([val])

    csvfile = 'Data/output_temp/brown_dominant_snps.csv'

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in brown_dominant:
            writer.writerow([val])


def recessive_check(upper, lower):
    data_bg = pd.read_csv('Data/output_temp/bg_snp_pct.csv')
    data_bg.sort_values(by=['pct_fm'], ascending=False, inplace=True)
    bg_fm_gtupper = data_bg.loc[data_bg['pct_fm'] >= upper]

    if len(bg_fm_gtupper) < int(0.01 * len(data_bg)):
        n_rows = int(len(data_bg) * 0.01)
        print(n_rows)
        bg_fm_gtupper = data_bg.head(n=n_rows)

    bg_fm_ltlower = data_bg.loc[data_bg['pct_fm'] <= lower]

    if len(bg_fm_ltlower) < int(0.01 * len(data_bg)):
        n_rows = int(len(data_bg) * 0.01)
        print(n_rows)
        bg_fm_ltlower = data_bg.tail(n=n_rows)
    print(bg_fm_ltlower)

    # Brown
    data_br = pd.read_csv('Data/output_temp/br_snp_pct.csv')
    data_br.sort_values(by=['pct_fm'], ascending=False, inplace=True)
    br_fm_gtupper = data_br.loc[data_br['pct_fm'] >= upper]

    if len(br_fm_gtupper) < int(0.01 * len(data_br)):
        n_rows = int(len(data_br) * 0.01)
        print(n_rows)
        br_fm_gtupper = data_br.head(n=n_rows)

    br_fm_ltlower = data_br.loc[data_br['pct_fm'] <= lower]
    if len(br_fm_ltlower) < int(0.01 * len(data_br)):
        n_rows = int(len(data_br) * 0.01)
        print(n_rows)
        br_fm_ltlower = data_bg.tail(n=n_rows)

    # ex: blue_recessive as those Rsids with  ((bg_fm_pct >= 75 and br_fm_pct <= 25))
    blue_recessive = set(bg_fm_gtupper['Rsid']).intersection(br_fm_ltlower['Rsid'])
    brown_recessive = set(br_fm_gtupper['Rsid']).intersection(bg_fm_ltlower['Rsid'])

    csvfile = 'Data/output_temp/blue_recessive_snps.csv'

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in blue_recessive:
            writer.writerow([val])

    csvfile = 'Data/output_temp/brown_recessive_snps.csv'

    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in brown_recessive:
            writer.writerow([val])


def combine_gene_snp(s):   # helper function for tree label
    """return part of Gene_info after colon """
    return s.rsplit(':', 1)[1] #  .rsplit() searches for the splitting string from the end of input string, and the second argument limits how many times it'll split to just once


def all_snps_on_gene(input_dir):
    blue_dominant = pd.read_csv(filepath_or_buffer='Data/output_temp/blue_dominant_snps.csv', header=None,
                                names=['Rsid'])
    blue_recessive = pd.read_csv(filepath_or_buffer='Data/output_temp/blue_recessive_snps.csv', header=None,
                                 names=['Rsid'])
    brown_dominant = pd.read_csv(filepath_or_buffer='Data/output_temp/brown_dominant_snps.csv', header=None,
                                 names=['Rsid'])
    brown_recessive = pd.read_csv(filepath_or_buffer='Data/output_temp/brown_recessive_snps.csv',
                                  header=None,
                                  names=['Rsid'])
    combined = brown_dominant['Rsid'].tolist() + blue_dominant['Rsid'].tolist() + blue_recessive['Rsid'].tolist() + \
               brown_recessive['Rsid'].tolist()
    print("Reading final file Blue Green...")
    data_bg = pd.read_csv(os.path.join(input_dir, 'preprocessed_Blue_Green.csv'))

    #### I don't think we use this
    pheno = pd.read_csv('Data/final_eyecolor_phenotypes.csv')

    blue_genes = data_bg.loc[data_bg['Rsid'].isin(combined),'Gene_info']
    blue_df= data_bg.loc[data_bg['Gene_info'].isin(blue_genes)]
    blue_df.drop_duplicates(subset='Rsid', inplace=True)

    # LATER: SHOULD DO THIS ONCE ONLY, IN STEP 1, NOT IN STEP 2 ML, which is separately for blue and brown
    # rename SNP so as to carry gene name: we want part of Gene_info after colon concat with Rsid
    gene_SNP_blue =  blue_df['Gene_info'].map(combine_gene_snp) + blue_df['Rsid']
    print(gene_SNP_blue)
    blue_df['Rsid'] = gene_SNP_blue

    blue_df.drop(labels='Gene_info', axis=1, inplace=True)

    blue_df_t = blue_df.transpose()
    blue_df_t = blue_df_t.rename(columns=blue_df_t.iloc[0])
    blue_df_t = blue_df_t[1:]
    blue_df_t['user_id'] = blue_df_t.index

    blue_df_t['user_id'] = blue_df_t['user_id'].apply(pd.to_numeric)

    print("Reading final file Brown...")
    data_br = pd.read_csv(os.path.join(input_dir, 'preprocessed_Brown.csv'))

    brown_genes = data_br.loc[data_br['Rsid'].isin(combined), 'Gene_info']

    brown_df = data_br.loc[data_br['Gene_info'].isin(brown_genes)]
    brown_df.drop_duplicates(subset='Rsid', inplace=True)

    # LATER: SHOULD DO THIS ONCE ONLY, IN STEP 1, NOT IN STEP 2 ML, which is separately for blue and brown
    # rename SNP so as to carry gene name: we want part of Gene_info after colon concat with Rsid
    gene_SNP_brown=  brown_df['Gene_info'].map(combine_gene_snp) + brown_df['Rsid']
    print(gene_SNP_brown)
    brown_df['Rsid'] = gene_SNP_brown

    brown_df.drop(labels='Gene_info', axis=1, inplace=True)

    brown_df_t = brown_df.transpose()
    brown_df_t = brown_df_t.rename(columns=brown_df_t.iloc[0])
    brown_df_t = brown_df_t[1:]
    brown_df_t['user_id'] = brown_df_t.index

    brown_df_t['user_id'] = brown_df_t['user_id'].apply(pd.to_numeric)
    blue_df_t['phenotype'] = 'Blue'
    brown_df_t['phenotype'] = 'Brown'
    final_dataset=pd.concat([blue_df_t,brown_df_t])

    #
    final_dataset.to_csv("Data/output_intermediate/final_dataset.csv", index=False)
    return "Data/output_intermediate/final_dataset.csv"
