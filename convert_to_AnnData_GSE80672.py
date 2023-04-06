import pandas as pd
import numpy as np
from pathlib import Path
from meth import tools as mtools
import scanpy as sc
import sys

coverage = int(sys.argv[1])
bin_by = int(sys.argv[2])

cwd = Path.cwd()
project_dir = Path('data/GSE80672')
project_id = 'GSE80672'
project_samples_dir = 'GSE80672_RAW'

soft_path = Path.joinpath(cwd, project_dir, '{}_family.soft.gz'.format(project_id))
metainfo_path = '1-s2.0-S1550413117301687-mmc2.xlsx'
gl_metainfo_path = Path.joinpath(project_dir, metainfo_path)

df_meta = pd.read_excel(gl_metainfo_path, skiprows=3, index_col=0)
to_category = ['Sex', 'Strain/Condition', 'Used in Subset 1/2', 'Prepared for Sequencing ']
df_meta.reset_index(inplace=True)
df_meta['Sample Name'] = df_meta['Sample Name'].apply(str.upper)
df_meta.set_index('Sample Name', inplace=True)
df_meta[to_category] = df_meta[to_category].astype("category")
df_meta["Age"][df_meta["Age"] == '2.5 (at isolation)'] = 2.5
df_meta["Age"] = df_meta["Age"].astype(np.float16)
df_meta.drop(["Aligned PE Reads",], axis=1, inplace=True)

data_path = Path.joinpath(cwd, project_dir, "{}_coverage_{}_binned_by_{}.csv".format(project_id, coverage, bin_by))

# df_coverage = mtools.get_methylation_df(data_path, typ='coverage')
df_percent = mtools.get_methylation_df(data_path, typ='percents')

ad = sc.AnnData(df_percent)
ad.var = pd.concat((ad.var, df_meta.loc[ad.var.index,:]), axis=1)
ad.X = ad.X.astype(np.float16) / 100.
ad.write_h5ad(Path.joinpath(project_dir, "{}_coverage_{}_binned_by_{}.h5ad".format(project_id, coverage, bin_by)))