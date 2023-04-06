import pandas as pd
import numpy as np
from collections import defaultdict
from pyliftover import LiftOver
import os
import sys

lo_mm9_to_mm10 = LiftOver(os.path.join(os.path.dirname(__file__),  'mm9ToMm10.over.chain'))
lo_mm10_to_mm39 = LiftOver(os.path.join(os.path.dirname(__file__),  'mm10ToMm39.over.chain'))
lo_mm10_to_hg19 = LiftOver(os.path.join(os.path.dirname(__file__),  'mm10ToHg19.over.chain'))
lo_mm10_to_hg38 = LiftOver(os.path.join(os.path.dirname(__file__),  'mm10ToHg38.over.chain'))

ncbi_chromosomes = ["NC_000067.6",
                        "NC_000068.7",
                        "NC_000069.6",
                        "NC_000070.6",
                        "NC_000071.6",
                        "NC_000072.6",
                        "NC_000073.6",
                        "NC_000074.6",
                        "NC_000075.6",
                        "NC_000076.6",
                        "NC_000077.6",
                        "NC_000078.6",
                        "NC_000079.6",
                        "NC_000080.6",
                        "NC_000081.6",
                        "NC_000082.6",
                        "NC_000083.6",
                        "NC_000084.6",
                        "NC_000085.6",
                        "NC_000086.7",
                        "NC_000087.7",
                        ]


def def_value():
    return "chrNone"

chromosomes = ["chr{}".format(x) for x in range(1, 20)] + ["chrX", "chrY"]
map_ncbi_to_chromosomes_dict = defaultdict(def_value, dict(zip(ncbi_chromosomes, chromosomes)))
def split_to_coverage_and_percentage(df):
    df_percentage = df.iloc[:,::2]
    df_coverage = df.iloc[:,1::2]
    return df_percentage, df_coverage

def map_ncbi_to_chromosomes(ncbi_id):
    chr_id = map_ncbi_to_chromosomes_dict[ncbi_id]
    return chr_id

def convert_cpg_refseq_id_to_chr_pos(ncbi_id):
    res = ncbi_id.replace(':', '').split('|')[-2:]
    res[0] = map_ncbi_to_chromosomes(res[0])
    if res[0] == 'chrNone':
        return np.nan
    else:
        return "_".join((res[0], res[1]))

def transform_indices_from_refseq_to_chr_pos(df):
    df.reset_index(inplace=True)
    df['index'] = df['index'].apply(convert_cpg_refseq_id_to_chr_pos)
    df = df[~df['index'].isna()]
    df.set_index('index', inplace=True)
    return df

def remove_technical_info_from_sample_id(df):
    df.columns = ["_".join(x.split('_')[:-1]) for x in df.columns]
    return df

def get_methylation_df(data_path, typ='percents'):
    cols = pd.read_csv(data_path, sep='\t', nrows=0).columns
    if typ == 'coverage':
        df = pd.read_csv(data_path, sep='\t', index_col=0, usecols=cols[::2])
        df = remove_technical_info_from_sample_id(df)
    elif typ == 'percents':
        df = pd.read_csv(data_path, sep='\t', index_col=0, usecols=[cols[0]] + cols[1::2].tolist())
        df = remove_technical_info_from_sample_id(df)
    else:
        df = pd.read_csv(data_path, sep='\t', index_col=0)
    return df

def convert_cpg(ch, pos, conversion='mm9_to_mm10'):
    if conversion == 'mm9_to_mm10':
        ch_new, pos_new, _, _ = lo_mm9_to_mm10.convert_coordinate(ch, pos)[0]
    elif conversion == 'mm10_to_mm39':
        ch_new, pos_new, _, _ = lo_mm10_to_mm39.convert_coordinate(ch, pos)[0]
    elif conversion == 'mm10_to_hg19':
        ch_new, pos_new, _, _ = lo_mm10_to_hg19.convert_coordinate(ch, pos)[0]
    elif conversion == 'mm10_to_hg38':
        ch_new, pos_new, _, _ = lo_mm10_to_hg38.convert_coordinate(ch, pos)[0]
    return ch_new, pos_new

def convert_between_assemblies(cpg, conversion='mm9_to_mm10', format_from='chr_pos', format_to='chr_pos'):
    if cpg == "None":
        return "None"
    try:
        if format_from == 'chr_pos':
            ch, pos = cpg.split('_')
            pos = int(pos)
            pos_st = int(pos)
            pos_end = int(pos) + 1
        elif format_from == 'chr:pos-pos':
            ch = cpg.split(':')[0]
            pos_st, pos_end = cpg.split(':')[1].split('-')
            pos_st, pos_end = int(pos_st), int(pos_end)
        ch_new, pos_new_st = convert_cpg(ch, pos_st, conversion=conversion)
        ch_new, pos_new_end = convert_cpg(ch, pos_end, conversion=conversion)
        if pos_new_end < pos_new_st:
            pos_new_end, pos_new_st = pos_new_st, pos_new_end
        if format_to == 'chr_pos':
            return '_'.join([ch_new, str(pos_new_st)])
        elif format_to == 'chr:pos-pos':
            return '-'.join([':'.join([ch_new, str(pos_new_st)]), str(pos_new_end)])
    except:
        return "None"

def convert_cpgs_notations(cpg):
    try:
        if '_' in cpg:
            ch, pos = cpg.split('_')
            pos_st = int(pos)
            pos_end = int(pos) + 1
            return '-'.join([':'.join([ch, str(pos_st)]), str(pos_end)])
        elif ':' in cpg:
            ch = cpg.split(':')[0]
            pos_st, pos_end = cpg.split(':')[1].split('-')
            pos_st, pos_end = int(pos_st), int(pos_end)
            return '_'.join([ch, str(pos_st)])
    except:
        return "None"

def transform_indices_between_assemblies(df, conversion='mm9_to_mm10'):
    df.reset_index(inplace=True)
    df['index'] = df['index'].apply(lambda x: convert_between_assemblies(x, conversion=conversion))
    df = df[df['index'] != 'None']
    df.set_index('index', inplace=True)
    return df

def bin_cpgs_by_thousands(cpg, bin_by=1000):
    if cpg == "None":
        return "None"
    try:
        ch, pos = cpg.split('_')
        pos = int(pos)
        ch_new = ch
        pos_new = (pos // bin_by) * bin_by
    except:
        return "None"
    return '_'.join([ch_new, str(pos_new)])

def get_binned_cpgs(df, bin_by=1000):
    df.reset_index(inplace=True)
    df['index'] = df['index'].apply(lambda x: bin_cpgs_by_thousands(x, bin_by=bin_by))
    df = df[df['index'] != 'None']
    df = df.groupby(['index']).sum()
    return df

