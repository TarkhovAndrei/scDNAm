import pandas as pd
import os
import sys
import pwd
from pathlib import Path
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map
import meth.tools as mtools
from scipy import sparse as sp
import scanpy as sc
import numpy as np

cwd = Path.cwd()
project_dir = Path('data/GSE80672')
project_id = 'GSE80672'
project_samples_dir = 'GSE80672_RAW'
path_samples = Path.joinpath(project_dir, project_samples_dir)
gl_path_samples = Path.joinpath(cwd, path_samples)
gl_project_dir = Path.joinpath(cwd, project_dir)
all_samples = os.listdir(gl_path_samples)
print(len(all_samples))
df_glob = None

coverage = int(sys.argv[1])
bin_by = int(sys.argv[2])
# coverage_binned = int(sys.argv[3])

def read_raw_sample_file(path_sample):
    df = pd.read_csv(path_sample, sep='\t', index_col=0)
    df = df[df.iloc[:,1] >= coverage]
    gsm = ("_".join(path_sample.name.split('.')[0].split('_')[1:])).upper()
    df.columns = ['{}_percentage'.format(gsm), '{}_coverage'.format(gsm)]
    df = mtools.transform_indices_from_refseq_to_chr_pos(df)
    #df = mtools.transform_indices_between_assemblies(df, conversion="mm10_to_mm39")
    df['{}_percentage'.format(gsm)] = df['{}_percentage'.format(gsm)].values * df['{}_coverage'.format(gsm)].values
    df = mtools.get_binned_cpgs(df, bin_by=bin_by)
    df['{}_percentage'.format(gsm)] = 1. * df['{}_percentage'.format(gsm)].values / df['{}_coverage'.format(gsm)].values
    df.dropna(axis=0, how='any', inplace=True)
    an = sc.AnnData(df)
    an.X = sp.csr_matrix(np.nan_to_num(an.X.astype(np.float32), copy=False))
    # dfs = pd.DataFrame({'{}_percentage'.format(gsm): pd.arrays.SparseArray(df['{}_percentage'.format(gsm)].values, fill_value=np.nan),
    #                     '{}_coverage'.format(gsm): pd.arrays.SparseArray(df['{}_coverage'.format(gsm)].values, fill_value=np.nan),
    #                     }, index=df.index)
    # return (dfs)
    return (an)

#    return (df)

def main(**kwargs):

    sample_path_tuples = [Path.joinpath(gl_path_samples, file) for file in all_samples]
    results = process_map(read_raw_sample_file, sample_path_tuples,
                                            max_workers=12,
                                            chunksize=1)
    # df_glob = pd.concat(results, axis=1)
    # df_glob.to_csv(Path.joinpath(gl_project_dir, "{}_coverage_{}_binned_by_{}.csv".format(project_id, coverage, bin_by)), sep='\t')

    an_glob = sc.concat(results, axis=1, join='outer')
    an_glob.write_h5ad(Path.joinpath(project_dir, "{}_coverage_{}_binned_by_{}.h5ad".format(project_id, coverage, bin_by)))

if __name__ == '__main__':

    main()