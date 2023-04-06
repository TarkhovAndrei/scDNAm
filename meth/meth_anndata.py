from . import analytic_tools as anatools
import scanpy as sc
import numpy as np
from pathlib import Path
from scipy.special import logit
import pandas as pd
import scipy.sparse as sp


class AnnDataSet(object):
    def __init__(self, **kwargs):
        self.paths = kwargs.get('paths', {})
        self.is_scDNAm = kwargs.get('is_scDNAm', True)
        self.dnames = self.paths.keys()
        self.dsets_red = {}
        self.dsets = self.download_dsets()
        self.chosen_genes = self.find_common_genes()
        self.add_var_mask(self.chosen_genes, var_mask_name='common_cpgs')
        self.add_obs_mask(obs_mask_name='full')

        # self.add_obs_mask(obs_mask_name='age>20')
        # self.add_obs_mask(obs_mask_name='age<20')
        # self.bin_age(by='Age', to='Age_bin', N_age_bins=20)
        # self.bin_age(by='Age', to='Age_bin_rough', N_age_bins=3)

    def dump_datasets(self, dump_key='_dump_0'):
        for key in self.dnames:
            sc.AnnData(X=self.dsets_red[key].X,
                       var=self.dsets_red[key].var,
                       obs=self.dsets_red[key].obs).write_h5ad(self.paths[key].as_posix().replace('.h5ad', dump_key + '.h5ad'))

    def download_dsets(self, dnames=None):
        dsets = {}
        dnames = self.dnames if dnames is None else dnames
        for key in dnames:
            h5ad_tmp = sc.read_h5ad(self.paths[key])
            if h5ad_tmp.shape[0] > h5ad_tmp.shape[1]:
                dsets[key] = h5ad_tmp.T
            else:
                dsets[key] = h5ad_tmp
        return dsets

    def find_common_genes(self, subset=None):
        if subset is None:
            subset = self.dnames

        chosen_genes = None

        for key in subset:
            chosen_genes = set(self.dsets[key].var.index.tolist()) if chosen_genes is None else chosen_genes & set(
                self.dsets[key].var.index.tolist())
            print(f"Genes before {len(self.dsets[key].var.index)}, and after merging {key}: {len(chosen_genes)}")

        print(f"Common genes number: {len(chosen_genes) if chosen_genes is not None else 0}")
        return np.sort(list(chosen_genes)) if chosen_genes is not None else None

    def add_var_mask(self, var_mask=None, var_mask_name=None):
        if var_mask is None:
            var_mask = self.chosen_genes
            var_mask_name = 'common_cpgs'

        for key in self.dnames:
            self.dsets[key].var[f'_{var_mask_name}_binary'] = False
            self.dsets[key].var.loc[var_mask, f'_{var_mask_name}_binary'] = True
            self.dsets[key].var[f'_{var_mask_name}_order'] = -1
            self.dsets[key].var.loc[var_mask, f'_{var_mask_name}_order'] = np.arange(len(var_mask)).astype(int)
            com, idx, jdx = np.intersect1d(self.dsets[key].var.index, var_mask, return_indices=True)
            self.dsets[key].uns[f'_var_order_{var_mask_name}'] = idx

        self.var_mask_name = var_mask_name

    def add_obs_mask(self, obs_mask_name=None, obs_mask_function=None):
        if obs_mask_name is None:
            obs_mask_name = 'full'
        self.obs_mask_name = obs_mask_name
        for key in self.dnames:
            if obs_mask_name == 'age>20':
                obs_mask = self.dsets[key].obs['Age'] > 20.
            elif obs_mask_name == 'age<20':
                obs_mask = self.dsets[key].obs['Age'] <= 20.
            elif obs_mask_name == 'only_Petkovich_C57BL6':
                obs_mask = self.dsets[key].obs['Strain/Condition'] == 'C57BL/6'
            elif obs_mask_name == 'full':
                obs_mask = np.zeros(self.dsets[key].obs.shape[0]) == 0
            elif obs_mask_name == 'not_0.5_strict':
                x = self.dsets_red[key].X.copy()
                obs_mask = np.sum(np.abs(x - 0.5) > 0.3, axis=1) / x.shape[1] > 0.9
            elif obs_mask_name == 'not_0.5_strict_sp':
                x = np.array(self.dsets_red[key].X.copy().todense())
                obs_mask = np.sum(np.abs(x - 0.5) > 0.3, axis=1) / x.shape[1] > 0.9
                obs_mask[1::2] = obs_mask[::2]
            elif obs_mask_name == 'not_0.5':
                x = self.dsets_red[key].X.copy()
                obs_mask = np.sum(np.abs(x - 0.5) > 0.3, axis=1) / x.shape[1] > 0.5
            elif obs_mask_name == 'not_0.5_sp':
                x = np.array(self.dsets_red[key].X.copy().todense())
                obs_mask = np.sum(np.abs(x - 0.5) > 0.3, axis=1) / x.shape[1] > 0.5
                obs_mask[1::2] = obs_mask[::2]
            elif (obs_mask_name == 'coverage') or (obs_mask_name == 'percentage'):
                if 'Data Type' not in self.dsets[key].obs.columns:
                    obs_mask = np.zeros(self.dsets[key].obs.shape[0]) == 0
                else:
                    obs_mask = self.dsets[key].obs['Data Type'].values == obs_mask_name
            else:
                obs_mask = obs_mask_function(self.dsets[key].obs)
            self.dsets[key].obs[f'_{obs_mask_name}_binary'] = False
            self.dsets[key].obs.loc[obs_mask, f'_{obs_mask_name}_binary'] = True
            self.dsets[key].uns[f'_obs_binary_{obs_mask_name}'] = self.dsets[key].obs[f'_{obs_mask_name}_binary']
            self.dsets[key].uns[f'_obs_order_{obs_mask_name}'] = \
            np.nonzero(self.dsets[key].obs[f'_{obs_mask_name}_binary'].values)[0]


    def bin_age(self, by='Age', to='Age_bin', N_age_bins=10):
        for key in self.dnames:
            self.dsets[key].obs[to] = np.nan
            idx_nonan = ~self.dsets[key].obs[by].isna()
            self.dsets[key].obs.loc[idx_nonan, to] = pd.cut(self.dsets[key].obs.loc[idx_nonan, by],
                                                            [np.min(self.dsets[key].obs.loc[
                                                                        idx_nonan, by].values) - 2e-3 + i * (np.max(
                                                                self.dsets[key].obs.loc[
                                                                    idx_nonan, by].values) + 4e-3 - np.min(
                                                                self.dsets[key].obs.loc[
                                                                    idx_nonan, by].values)) / N_age_bins
                                                             for i in range(N_age_bins + 1)], include_lowest=True)

    def get_subsets_copy(self, obs_mask_name=None, var_mask_name=None):
        dsets_red = {}
        for key in self.dnames:
            dsets_red[key] = sc.AnnData(self.dsets[key].X[
                                            self.dsets[key].uns[f'_obs_binary_{obs_mask_name}'], self.dsets[key].uns[
                                                f'_var_order_{var_mask_name}']],
                                        self.dsets[key].obs.iloc[self.dsets[key].uns[f'_obs_binary_{obs_mask_name}'],
                                        :],
                                        self.dsets[key].var.iloc[self.dsets[key].uns[f'_var_order_{var_mask_name}'], :]
                                        )
        return dsets_red

    def get_subsets_view(self, obs_mask_name=None, var_mask_name=None):
        dsets_red = {}
        for key in self.dnames:
            dsets_red[key] = self.dsets[key][
                self.dsets[key].uns[f'_obs_order_{obs_mask_name}'], self.dsets[key].uns[f'_var_order_{var_mask_name}']]
            self.chosen_genes = dsets_red[key].var.index.copy()
        return dsets_red

    def update_dsets_view(self, obs_mask_name, var_mask_name):
        self.obs_mask_name = obs_mask_name
        self.var_mask_name = var_mask_name
        self.dsets_red = self.get_subsets_view(self.obs_mask_name, self.var_mask_name)


    def assign_var_masked_value(self, key, values, new_var_name=None):
        self.dsets[key].var[new_var_name] = np.nan
        self.dsets[key].var.loc[
            self.dsets[key].var.index[self.dsets[key].uns[f'_var_order_{self.var_mask_name}']], new_var_name] = values
        self.update_dsets_view(self.obs_mask_name, self.var_mask_name)

    def assign_obs_masked_value(self, key, values, new_var_name=None):
        self.dsets[key].obs[new_var_name] = np.nan
        self.dsets[key].obs.loc[
            self.dsets[key].obs.index[self.dsets[key].uns[f'_obs_order_{self.obs_mask_name}']], new_var_name] = values
        self.update_dsets_view(self.obs_mask_name, self.var_mask_name)

    def project_bioage(self, key_origin, bioage_name=None, obs_mask_name=None):
        if obs_mask_name is None:
            obs_mask_name = 'full'
        self.update_dsets_view(obs_mask_name, self.var_mask_name)
        b = self.dsets_red[key_origin].var[f'b_{bioage_name}'].values.reshape(-1, 1)
        x0 = self.dsets_red[key_origin].var[f'x0_{bioage_name}'].values.reshape(-1, 1)
        idx = np.abs(b) > 1e-4
        idx = idx.flatten()
        b[idx] = 1. / b[idx]
        # for key in self.dnames:
        Y = anatools.mask_0_1_tonan(self.dsets_red[key_origin].X.copy()).astype(np.float64)
        if ('logit' in bioage_name) and (isinstance(Y, np.ndarray)):
            Y = anatools.logit(Y)
        if isinstance(Y, np.ndarray):
            Y = np.matrix(Y)
            b = np.matrix(b.reshape(1,-1))
            x0 = np.matrix(x0.reshape(1,-1))

            Ynna = 1. * (~np.isnan(Y))
            rescale_for_nans = (1. * (~np.isnan(Y)[:, idx])).sum(axis=1)
            Y = np.nan_to_num(Y)
            ba = (np.multiply(np.multiply(Ynna[:, idx], b[:, idx]), Y[:, idx]).sum(axis=1) - np.multiply(
                np.multiply(Ynna[:, idx], x0[:, idx]), b[:, idx]).sum(axis=1)) / rescale_for_nans
            ba = np.array(ba).flatten()

#            ba = (np.multiply(np.multiply(~np.isnan(Y)[:, idx], Y[:, idx]), b[:,idx]).sum(axis=1) - np.multiply(np.multiply(~np.isnan(Y)[:, idx],x0[:,idx]), b[:,idx]).sum(axis=1)) / rescale_for_nans
        else:
            y = Y[::2,:]
            if np.nanmax(y.data) > 1.1:
                y.data /= 100.
            ycov = Y[1::2,:]
            b = sp.csr_matrix(b.reshape(1, -1))
            x0 = sp.csr_matrix(x0.reshape(1, -1))
            rescale_for_nans = (1. * (ycov[:,idx] > 0.5)).sum(axis=1)
            # ba = np.dot(np.nan_to_num((Y[idx, :] - x0[idx])).T, b[idx]).flatten() / rescale_for_nans.flatten()
            # b = ((Xcov > 0.5).multiply(X).multiply(tsp).sum(axis=axis) / number_nonans - np.multiply(X_mean,t_mean)) / t_var
            ba = ((1. * (ycov[:,idx] > 0.5)).multiply(b[:,idx]).multiply(y[:,idx]).sum(axis=1) - (1. * (ycov[:,idx] > 0.5)).multiply(x0[:,idx]).multiply(b[:,idx]).sum(axis=1)) / rescale_for_nans
            ba = np.array(ba).flatten()
            ba2 = np.zeros(ba.shape[0] * 2)
            ba2[::2] = ba.copy()
            ba2[1::2] = ba.copy()
            ba = ba2.copy()

        # rescale_for_nans = 1. * np.sum(~np.isnan(Y)[idx, :], axis=0)
        # ba = np.dot(np.nan_to_num((Y[idx, :] - x0[idx])).T, b[idx]).flatten() / rescale_for_nans.flatten()

        self.assign_obs_masked_value(key_origin, ba, new_var_name=f"{key_origin}_{bioage_name}")

    def Horvath_transform(self, t_ev):
        res = t_ev.copy()
        idx = res < 0
        res[idx] = 21. * np.exp(res[idx]) - 1.
        res[~idx] = 21. * res[~idx] + 20.
        return res

    def project_Horvath(self, key_origin=None, obs_mask_name=None):
        if key_origin is None:
            key_origins = self.dnames
        else:
            key_origins = [key_origin]
        if obs_mask_name is None:
            obs_mask_name = 'full'
        cwd = Path.cwd()
        clock_paths = {'Horvath_blood_clock': Path.joinpath(cwd, 'Horvath_blood_clock.csv'),
                       'Horvath_brain_clock': Path.joinpath(cwd, 'Horvath_brain_clock.csv'),
                       }
        clock_dfs = {}

        for clock in clock_paths.keys():
            clock_dfs[clock] = pd.read_csv(clock_paths[clock])
            cpgs = clock_dfs[clock]['CpGmarker'].values[1:]

            self.update_dsets_view('full', 'common_cpgs')
            cpgs_common = np.intersect1d(cpgs, self.chosen_genes)

            t0 = clock_dfs[clock]['CoefficientTraining'].values[0]
            clock_dfs[clock].drop(0, axis=0, inplace=True)
            clock_dfs[clock].set_index('CpGmarker', inplace=True)
            clock_dfs[clock] = clock_dfs[clock].loc[cpgs_common]
            betas = clock_dfs[clock]['CoefficientTraining'].values

            self.add_var_mask(cpgs_common, clock)

            for key_origin in key_origins:
                self.update_dsets_view(obs_mask_name, self.var_mask_name)
                Y = anatools.mask_0_1_tonan(self.dsets_red[key_origin].X.copy()).T.astype(np.float64)
                ba = t0 + np.dot(np.nan_to_num(Y).T, betas).flatten()
                ba_resc = self.Horvath_transform(ba)
                self.assign_obs_masked_value(key_origin, ba, new_var_name=f"{key_origin}_{clock}_raw")
                self.assign_obs_masked_value(key_origin, ba_resc, new_var_name=f"{key_origin}_{clock}")

                self.assign_var_masked_value(key_origin, betas, new_var_name=f'b_{clock}')
                self.assign_var_masked_value(key_origin, t0 / (1. * len(betas)) + 0. * betas,
                                             new_var_name=f'x0_{clock}')

    def project_Thompson(self, key_origin=None, obs_mask_name=None, var_mask_name='common_cpgs', suffix=''):
        if key_origin is None:
            key_origins = self.dnames
        else:
            key_origins = [key_origin]
        if obs_mask_name is None:
            obs_mask_name = 'full'
        cwd = Path.joinpath(Path.cwd(), 'clocks')
        clock_paths = {f'Thompson_multi_tissue{suffix}': Path.joinpath(cwd, f'Thompson_multi_tissue{suffix}.csv'),
                       f'Thompson_Liver_tissue{suffix}': Path.joinpath(cwd, f'Thompson_Liver_tissue{suffix}.csv'),
                       f'MeerClock{suffix}': Path.joinpath(cwd, f'MeerClock{suffix}.csv'),
                       f'PetkovichClock{suffix}': Path.joinpath(cwd, f'PetkovichClock{suffix}.csv'),
                       f'PetkovichOriginalClock{suffix}': Path.joinpath(cwd, f'PetkovichOriginalClock{suffix}.csv'),
                       f'ThompsonClock-all{suffix}': Path.joinpath(cwd, f'ThompsonClock-all{suffix}.csv'),
                       f'ThompsonClock-conserved{suffix}': Path.joinpath(cwd, f'ThompsonClock-conserved{suffix}.csv'),
                       f'WangLiver{suffix}': Path.joinpath(cwd, f'WangLiver.csv'),
                       }

        clock_dfs = {}

        for clock in clock_paths.keys():
            clock_dfs[clock] = pd.read_csv(clock_paths[clock])
            clock_dfs[clock].drop(0, axis=0, inplace=True)
            cpgs = clock_dfs[clock]['ID'].values
            clock_dfs[clock].set_index('ID', inplace=True)

            self.update_dsets_view('full', var_mask_name)
            cpgs_common = np.intersect1d(cpgs, self.chosen_genes)

            if clock == 'Thompson_Liver_tissue':
                t0 = 5.82
            elif clock == 'Thompson_multi_tissue':
                t0 = 234.64
            else:
                t0 = 0.0

            clock_dfs[clock] = clock_dfs[clock].loc[cpgs_common]
            betas = clock_dfs[clock]['Weight'].values

            self.add_var_mask(cpgs_common, clock)

            for key_origin in key_origins:
                # self.update_dsets_view(obs_mask_name, self.var_mask_name)
                self.update_dsets_view(obs_mask_name, clock)
                Y = anatools.mask_0_1_tonan(self.dsets_red[key_origin].X.copy()).astype(np.float64).toarray()
                ba = t0 + np.dot(np.nan_to_num(Y), betas).flatten()
                # ba_resc = self.Horvath_transform(ba)
                ba_resc = ba
                self.assign_obs_masked_value(key_origin, ba, new_var_name=f"{key_origin}_{clock}_raw")
                self.assign_obs_masked_value(key_origin, ba_resc, new_var_name=f"{key_origin}_{clock}")

                self.assign_var_masked_value(key_origin, betas, new_var_name=f'b_{clock}')
                self.assign_var_masked_value(key_origin, t0 / (1. * len(betas)) + 0. * betas,
                                             new_var_name=f'x0_{clock}')


    def find_aging_associations(self, source_dsets=None, target_dsets=None, obs_mask_name=None, var_mask_name=None, association_name=None):
        self.update_dsets_view(obs_mask_name, var_mask_name)
        if source_dsets is None:
            source_dsets = self.dsets
        if target_dsets is None:
            target_dsets = self.dsets

        for key in source_dsets:
            bioage_name = f"{key}_{obs_mask_name}_{var_mask_name}_{association_name}"
            Y = anatools.mask_0_1_tonan(self.dsets_red[key].X.copy()).astype(np.float64)
            if (association_name == 'logit') and (isinstance(Y, np.ndarray)):
                Y = anatools.logit(Y)
            age = self.dsets_red[key].obs['Age'].values.copy().astype(np.float64)
            r, p = anatools.fast_pearsonr(Y, age)
            b, x0 = anatools.shooting(Y, age)
            for key2apply in target_dsets:
                self.assign_var_masked_value(key2apply, r, new_var_name=f'r_{bioage_name}')
                self.assign_var_masked_value(key2apply, p, new_var_name=f'p_{bioage_name}')

                self.assign_var_masked_value(key2apply, b * 1. * (p > 0.05), new_var_name=f'b_{bioage_name}_non_sign')
                self.assign_var_masked_value(key2apply, x0 * 1. * (p > 0.05), new_var_name=f'x0_{bioage_name}_non_sign')

                self.project_bioage(key2apply, bioage_name=f'{bioage_name}_non_sign')

                self.assign_var_masked_value(key2apply, b, new_var_name=f'b_{bioage_name}')
                self.assign_var_masked_value(key2apply, x0, new_var_name=f'x0_{bioage_name}')

                self.project_bioage(key2apply, bioage_name=bioage_name)

                self.assign_var_masked_value(key2apply, b * 1. * (p < 0.05), new_var_name=f'b_{bioage_name}_sign')
                self.assign_var_masked_value(key2apply, x0 * 1. * (p < 0.05), new_var_name=f'x0_{bioage_name}_sign')

                self.project_bioage(key2apply, bioage_name=f'{bioage_name}_sign')

                self.assign_var_masked_value(key2apply, b * 1. * (p < 0.05 / 500000.),
                                             new_var_name=f'b_{bioage_name}_Bonf_sign')
                self.assign_var_masked_value(key2apply, x0 * 1. * (p < 0.05 / 500000.),
                                             new_var_name=f'x0_{bioage_name}_Bonf_sign')

                self.project_bioage(key2apply, bioage_name=f'{bioage_name}_Bonf_sign')

