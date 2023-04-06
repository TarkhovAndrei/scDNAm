import numpy as np
import pandas as pd
import scipy
import sparse as spa
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.special import entr
import h5py
from tqdm import tqdm, trange


def my_colormap(names, full = False, cmap = plt.rcParams['image.cmap'], **kwargs):
    """Return dict or list(if `full` = `True`) with colors, with respect to `names`"""
#     snames = np.sort(names)
#     uname = np.unique(snames)
#     uname = dict(zip(uname, np.arange(uname.shape[0])))
#
    names = np.asarray(names)
    uname = np.unique(names)
    if names.dtype == float or names.dtype == int:
        uname = dict(zip(uname, uname))
        vmin, vmax = names.min(), names.max()
    else:
        vmin, vmax = 0, uname.shape[0] -1
        uname = dict(zip(uname, np.arange(uname.shape[0])))
    vmin, vmax = kwargs.get('vnorm', (vmin, vmax))
    cNorm  = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,
            cmap=cmap)
    colorfunc = np.vectorize( scalarMap.to_rgba )
    if full:
        return np.r_[[scalarMap.to_rgba(uname[x]) for x in names]]
    else:
        return dict(zip(uname, np.r_[[scalarMap.to_rgba(uname[x]) for x in uname]] ))




def fast_pearsonr(X, t):
    if isinstance(X, np.ndarray):
        return fast_pearsonr_ndarray_mat(X, t, axis=0)
    else:
        x = X[::2, :]
        xcov = X[1::2, :]
        t = t[::2]
        if np.nanmax(x.data) > 1.1:
            x.data /= 100.
        return fast_pearsonr_sparse(x, xcov, t)

def fast_pearsonr_sparse(X, Xcov, t, axis=0):
    if type(X) is not np.float64:
        X = X.astype(np.float64)
    if type(Xcov) is not np.float64:
        Xcov = Xcov.astype(np.float64)
    if type(t) is not np.float64:
        t = t.astype(np.float64)
    tsp = sp.csr_matrix(t.reshape(-1,1))#, shape=(t.shape[0], 1))
    number_nonans = (1.0 * (1. * Xcov > 0.5)).sum(axis=axis)
    # TEST
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan

    X_mean = (1. * (Xcov > 0.5)).multiply(X).sum(axis=axis) /  number_nonans
    t_mean = (1. * (Xcov > 0.5)).multiply(tsp).sum(axis=axis) /  number_nonans

    r_nomin = ((1. * (Xcov > 0.5)).multiply(X).multiply((Xcov > 0.5).multiply(tsp)).sum(axis=axis) -
               (1. * (Xcov > 0.5)).multiply(tsp).multiply(X_mean).sum(axis=axis) -
               (1. * (Xcov > 0.5)).multiply(X).multiply(t_mean).sum(axis=axis) +
               (1. * (Xcov > 0.5)).multiply(X_mean).multiply(t_mean).sum(axis=axis)
              )

    r_denomin = np.sqrt(np.multiply(
               ((1. * (Xcov > 0.5)).multiply(X).multiply(X).sum(axis=axis) -
               (1. * (Xcov > 0.5)).multiply(X).multiply(2 * X_mean).sum(axis=axis) +
               (1. * (Xcov > 0.5)).multiply(np.power(X_mean,2)).sum(axis=axis)
              ),
              ((1. * (Xcov > 0.5)).multiply(tsp).multiply(tsp).sum(axis=axis) -
               (1. * (Xcov > 0.5)).multiply(tsp).multiply(2 * t_mean).sum(axis=axis) +
               (1. * (Xcov > 0.5)).multiply(np.power(t_mean,2)).sum(axis=axis)
                        )
               )
                       )

    r_denomin[np.abs(r_denomin) < 1e-4] = np.nan
    r = r_nomin/r_denomin
    r = np.array(r).flatten()
    dist = scipy.stats.beta(number_nonans/2. - 1, number_nonans/2. - 1, loc=-1, scale=2)
    # idx = ~np.isnan(r)
    # p = 0. * r + 1.
    # p[idx] = 2 * dist.cdf(-np.abs(r[idx]))
    p = 2 * dist.cdf(-np.abs(r))
    p = np.array(p).flatten()
    return (r, p)


def fast_pearsonr_ndarray_mat(X, t, axis=0):
    X = X.astype(np.float64)
    t = t.astype(np.float64)
    X = np.matrix(X)
    if axis == 0:
        t = np.matrix(t.reshape(-1, 1))
    else:
        t = np.matrix(t.reshape(1, -1))
    number_nonans = (1. * (~np.isnan(X))).sum(axis=axis)
    Xnna = 1. * (~np.isnan(X))
    X = np.nan_to_num(X)
    X_mean = np.multiply(Xnna, X).sum(axis=axis) / number_nonans
    t_mean = np.multiply(Xnna, t).sum(axis=axis) / number_nonans
    #
    # t_var = (np.multiply(np.multiply(~np.isnan(X), t), t).sum(axis=axis) - np.multiply(np.multiply(t_mean, t_mean),
    #                                                                                    number_nonans))
    # X_var = (np.multiply(np.multiply(~np.isnan(X), X), X).sum(axis=axis) - np.multiply(np.multiply(X_mean, X_mean),
    #                                                                                    number_nonans))

    r_nomin = (np.multiply(np.multiply(Xnna, X), t).sum(axis=axis) -
         np.multiply(np.multiply(Xnna, t), X_mean).sum(axis=axis) -
         np.multiply(np.multiply(Xnna, X), t_mean).sum(axis=axis) +
         np.multiply(np.multiply(Xnna, X_mean), t_mean).sum(axis=axis)
         )
    r_denomin = np.sqrt(np.multiply(
        (np.multiply(np.multiply(Xnna, X), X).sum(axis=axis) -
         np.multiply(np.multiply(Xnna, X), 2 * X_mean).sum(axis=axis) +
         np.multiply(np.multiply(Xnna, X_mean), X_mean).sum(axis=axis)
         ),
        (np.multiply(np.multiply(Xnna, t), t).sum(axis=axis) -
         np.multiply(np.multiply(Xnna, t), 2 * t_mean).sum(axis=axis) +
         np.multiply(np.multiply(Xnna, t_mean), t_mean).sum(axis=axis)
         )
    )
    )
    
    r = r_nomin/r_denomin
    dist = scipy.stats.beta(number_nonans / 2. - 1, number_nonans / 2. - 1, loc=-1, scale=2)
    p = 2 * dist.cdf(-np.abs(r))
    r = np.array(r).flatten()
    p = np.array(p).flatten()
    return (r, p)

def fast_pearsonr_ndarray(X, t):
    if type(X) is not np.float64:
        X = X.astype(np.float64)
    if type(t) is not np.float64:
        t = t.astype(np.float64)
    number_nonans = np.sum(~np.isnan(X), axis=1)
    t_matrix = np.repeat(t.reshape(1, -1), X.shape[0], axis=0)
    t_matrix[np.isnan(X)] = np.nan
    X_mean = np.nanmean(X, axis=1)
    r = np.nansum((X.T - X_mean).T * (t - np.nanmean(t)), axis=1)
    r /= np.sqrt(np.nansum((X.T - X_mean) ** 2, axis=0))
    r /= np.sqrt(np.nansum((t_matrix.T - np.nanmean(t_matrix, axis=1)) ** 2, axis=0))
    dist = scipy.stats.beta(number_nonans/2. - 1, number_nonans/2. - 1, loc=-1, scale=2)
    p = 2 * dist.cdf(-np.abs(r))
    return (r, p)

def shooting_sparse(X, Xcov, t, axis = 0):
    X = X.astype(np.float64)
    Xcov = Xcov.astype(np.float64)
    t = t.astype(np.float64)
    tsp = sp.csr_matrix(t.reshape(-1,1))
    number_nonans = (1.0 * (Xcov > 0.5)).sum(axis=axis)
    # TEST
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    X_mean = (1. * (Xcov > 0.5)).multiply(X).sum(axis=axis) /  number_nonans
    t_mean = (1. * (Xcov > 0.5)).multiply(tsp).sum(axis=axis) /  number_nonans
    t_var = ((1. * (Xcov > 0.5)).multiply(tsp).multiply(tsp).sum(axis=axis) / number_nonans - np.multiply(t_mean,t_mean))
    # TEST
    t_var[np.abs(t_var) < 1e-4] = np.nan
    b = ((1. * (Xcov > 0.5)).multiply(X).multiply(tsp).sum(axis=axis) / number_nonans - np.multiply(X_mean,t_mean)) / t_var
    x0 = X_mean - np.multiply(b, t_mean)
    b = np.array(b).flatten()
    x0 = np.array(x0).flatten()
    return (b, x0)

def shooting(X, t, axis=0):
    if isinstance(X, np.ndarray):
        return shooting_ndarray_mat(X, t, axis=0)
    else:
        x = X[::2, :]
        # print(np.nanmax(x.data))
        if np.nanmax(x.data) > 1.1:
            x.data /= 100.
        xcov = X[1::2, :]
        t = t[::2]
        return shooting_sparse(x, xcov, t)


def shooting_ndarray(X, t):
    if type(X) is not np.float64:
        X = X.astype(np.float64)
    if type(t) is not np.float64:
        t = t.astype(np.float64)
    number_nonans = np.sum(~np.isnan(X), axis=1)
    t_matrix = np.repeat(t.reshape(1, -1), X.shape[0], axis=0)
    t_matrix[np.isnan(X)] = np.nan
    X_mean = np.nanmean(X, axis=1)
    t_mean = np.nanmean(t_matrix, axis=1)
    t_std = np.nanstd(t_matrix, axis=1)
    b = (np.nansum(X * t_matrix, axis=1) / number_nonans - X_mean * t_mean) / (t_std ** 2)
    x0 = X_mean - b * t_mean
    return (b, x0)


def shooting_ndarray_mat(X, t, axis=0):
    if type(X) is not np.float64:
        X = X.astype(np.float64)
    if type(t) is not np.float64:
        t = t.astype(np.float64)
    X = np.matrix(X)
    if axis == 0:
        t = np.matrix(t.reshape(-1,1))
    else:
        t = np.matrix(t.reshape(1,-1))
    Xnna = 1. * (~np.isnan(X))
    number_nonans = np.sum(1. * (~np.isnan(X)), axis=axis)
    X = np.nan_to_num(X)
    X_mean = np.multiply(Xnna, X).sum(axis=axis) / number_nonans
    t_mean = np.multiply(Xnna, t).sum(axis=axis) / number_nonans
    t_var = (np.multiply(np.multiply(Xnna, t), t).sum(axis=axis) / number_nonans - np.multiply(t_mean, t_mean))
    b = (np.multiply(np.multiply(Xnna, X), t).sum(axis=axis) / number_nonans - np.multiply(X_mean, t_mean)) / (t_var)
    x0 = X_mean - np.multiply(b, t_mean)
    b = np.array(b).flatten()
    x0 = np.array(x0).flatten()
    return (b, x0)



def sparse_cov_mat(X, Xcov, axis=0):
    if type(X) is not np.float64:
        X = X.astype(np.float64)
    if type(Xcov) is not np.float64:
        Xcov = Xcov.astype(np.float64)
    if np.nanmax(X.data) > 1.1:
        X.data /= 100.
    xcov = 1. * (Xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    X_mean = xcov.multiply(X).sum(axis=axis) /  number_nonans
    cov_norm = spa.tensordot(xcov, xcov, axes=([axis], [axis]))
    cov = (spa.tensordot(X.multiply(xcov), X.multiply(xcov), axes=([axis], [axis])) -
          spa.tensordot(xcov.multiply(X_mean), xcov.multiply(X), axes=([axis], [axis])) -
          spa.tensordot(xcov.multiply(X), xcov.multiply(X_mean), axes=([axis], [axis])) +
          spa.tensordot(xcov.multiply(X_mean), xcov.multiply(X_mean), axes=([axis], [axis])))
    cov = np.array(cov.todense())
    cov_norm = np.array(cov_norm.todense())
    cov_mat = cov / cov_norm
    return cov_mat

def sparse_corr_mat(X, Xcov, axis=0):
    if type(X) is not np.float64:
        X = X.astype(np.float64)
    if type(Xcov) is not np.float64:
        Xcov = Xcov.astype(np.float64)
    if np.nanmax(X.data) > 1.1:
        X.data /= 100.
    xcov = 1. * (Xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    X_mean = xcov.multiply(X).sum(axis=axis) /  number_nonans
    cov_norm = spa.tensordot(xcov, xcov, axes=([axis], [axis]))
    cov = (spa.tensordot(X.multiply(xcov), X.multiply(xcov), axes=([axis], [axis])) -
          spa.tensordot(xcov.multiply(X_mean), xcov.multiply(X), axes=([axis], [axis])) -
          spa.tensordot(xcov.multiply(X), xcov.multiply(X_mean), axes=([axis], [axis])) +
          spa.tensordot(xcov.multiply(X_mean), xcov.multiply(X_mean), axes=([axis], [axis])))
    cov_denomin = (xcov.multiply(X).multiply(X).sum(axis=axis) -
           xcov.multiply(np.power(X_mean,2)).sum(axis=axis))
    cov_denomin = 1./np.sqrt(cov_denomin)
    cov_denomin = np.array(cov_denomin)
    cov = np.array(cov.todense())
    cov_norm = np.array(cov_norm.todense())
    cov_mat = cov * cov_denomin
    cov_mat = cov_denomin.T * cov_mat
    return cov_mat


def sparse_cov_mat_by_batch(X, Xcov, axis=0, idx=None, jdx=None):
    # if type(X) is not np.float64:
    #     X = X.astype(np.float64)
    # if type(Xcov) is not np.float64:
    #     Xcov = Xcov.astype(np.float64)
    if np.nanmax(X.data) > 1.1:
        X.data /= 100.
    if idx is None:
        idx = np.arange(X.shape[1])
    if jdx is None:
        jdx = np.arange(X.shape[1])
    xcov = 1. * (Xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    X_mean = xcov.multiply(X).sum(axis=axis) / number_nonans
    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis]))
    cov = (spa.tensordot(X[:, idx].multiply(xcov[:, idx]), X[:, jdx].multiply(xcov[:, jdx]), axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx].multiply(X_mean[:, idx]), xcov[:, jdx].multiply(X[:, jdx]),
                         axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx].multiply(X[:, idx]), xcov[:, jdx].multiply(X_mean[:, jdx]),
                         axes=([axis], [axis])) +
           spa.tensordot(xcov[:, idx].multiply(X_mean[:, idx]), xcov[:, jdx].multiply(X_mean[:, jdx]),
                         axes=([axis], [axis])))
    # cov = np.array(cov.todense())
    # cov_norm = np.array(cov_norm.todense())
    cov_mat = cov / cov_norm
    # return cov_mat
    return np.array(cov_mat.todense())

def sparse_corr_mat_by_batch(X, Xcov, axis=0, idx=None, jdx=None):
    if type(X) is not np.float64:
        X = X.astype(np.float64)
    if type(Xcov) is not np.float64:
        Xcov = Xcov.astype(np.float64)
    if np.nanmax(X.data) > 1.1:
        X.data /= 100.
    if idx is None:
        idx = np.arange(X.shape[1])
    if jdx is None:
        jdx = np.arange(X.shape[1])
    xcov = 1. * (Xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    X_mean = xcov.multiply(X).sum(axis=axis) / number_nonans
    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis]))
    cov = (spa.tensordot(X[:, idx].multiply(xcov[:, idx]), X[:, jdx].multiply(xcov[:, jdx]), axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx].multiply(X_mean[:, idx]), xcov[:, jdx].multiply(X[:, jdx]),
                         axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx].multiply(X[:, idx]), xcov[:, jdx].multiply(X_mean[:, jdx]),
                         axes=([axis], [axis])) +
           spa.tensordot(xcov[:, idx].multiply(X_mean[:, idx]), xcov[:, jdx].multiply(X_mean[:, jdx]),
                         axes=([axis], [axis])))
    cov_denomin = (xcov.multiply(X).multiply(X).sum(axis=axis) -
                   xcov.multiply(np.power(X_mean, 2)).sum(axis=axis))
    cov_denomin = 1. / np.sqrt(cov_denomin)
    cov_denomin = np.array(cov_denomin)
    cov = np.array(cov.todense())
    cov_norm = np.array(cov_norm.todense())
    cov_mat = cov * cov_denomin[:, jdx]
    cov_mat = cov_denomin[:, idx].T * cov_mat
    return cov_mat, 1./cov_norm

def cross_entropy_by_batch(x_, xcov_, axis=0, order=2, idx=None, jdx=None):
    x = x_.copy()
    xcov = xcov_.copy()
    x[x>=0.5] = 1.
    x[x<0.5] = 0.
    x[xcov < 0.5] = np.nan
    if idx is None:
        idx = np.arange(x.shape[1])
    if jdx is None:
        jdx = np.arange(x.shape[1])
    x = sp.csr_matrix(x)
    xcov = sp.csr_matrix(xcov)
    probs = np.zeros(np.power(2, order))
    ents = sp.csr_matrix(np.zeros((len(idx), len(jdx))))
    xcov = 1. * (xcov > 0.5)
    cov_norm = spa.tensordot(xcov[:,idx], xcov[:,jdx], axes=([axis], [axis])).asformat('csr').to_scipy_sparse()
    cov_norm.data = 1./ cov_norm.data
    for i in range(probs.shape[0]):
            x1 = i >> 1
            x2 = i % 2
            if x1 == 1:
                left = (xcov[:,idx]).multiply(1. * (x[:,idx] == 1))
            else:
                left = (xcov[:,idx]).multiply(1. * (x[:,idx] != 1))
            if x2 == 1:
                right = (xcov[:,jdx]).multiply(1. * (x[:,jdx] == 1))
            else:
                right = (xcov[:,jdx]).multiply(1. * (x[:,jdx] != 1))
            cov = spa.tensordot(left, right, axes=([axis], [axis])).asformat('csr').to_scipy_sparse()
            res = cov.multiply(cov_norm)
            # res = entr(res)
            log2res = res.copy()
            log2res.data = entr(log2res.data)
            ents += log2res / np.log(2)
            # log2res = res.copy()
            # log2res.data = np.log2(log2res.data)
            # ents += -(res).multiply(log2res)
    return ents, cov_norm


def cross_entropy_by_batch_via_covarmatr(x_, xcov_, axis=0, order=2, idx=None, jdx=None, ignore_nans=True):
    x = x_.copy()
    xcov = xcov_.copy()
    x[x >= 0.5] = 1.
    x[x < 0.5] = 0.
    if idx is None:
        idx = np.arange(x.shape[1])
    if jdx is None:
        jdx = np.arange(x.shape[1])
    Qg1g2 = sp.csr_matrix(sparse_cov_mat_by_batch(x, xcov, idx=idx, jdx=jdx))
    # x[xcov < 0.5] = 0.#np.nan
    xcov = 1. * (xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    x_mean = xcov.multiply(x).sum(axis=axis) / number_nonans
    x_g1 = sp.csr_matrix(np.broadcast_to(x_mean[:, idx].T, (len(idx), len(jdx))))
    x_g2 = sp.csr_matrix(np.broadcast_to(x_mean[:, jdx], (len(idx), len(jdx))))

    x_g1g2 = spa.tensordot((xcov[:, idx]).multiply(x[:, idx]), xcov[:, jdx], axes=([axis], [axis])).asformat(
        'csr').to_scipy_sparse()
    x_g2g1 = spa.tensordot((xcov[:, idx]), xcov[:, jdx].multiply(x[:, jdx]), axes=([axis], [axis])).asformat(
        'csr').to_scipy_sparse()
    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis])).asformat('csr').to_scipy_sparse()
    cov_norm.data = 1. / cov_norm.data

    x_g1g2 = x_g1g2.multiply(cov_norm)
    x_g2g1 = x_g2g1.multiply(cov_norm)

    xg1xg2 = (Qg1g2 +
              sp.csr_matrix(x_g1.multiply(x_g2g1)) +
              sp.csr_matrix(x_g2.multiply(x_g1g2)) -
              sp.csr_matrix(x_g1.multiply(x_g2)))
    ents = sp.lil_matrix(np.zeros((len(idx), len(jdx))))
    # ents = sp.csr_matrix(np.zeros((len(idx), len(jdx))))

    for i in range(4):
        if i == 0:
            res = xg1xg2 - x_g1g2 - x_g2g1
            res.data += 1.
        elif i == 1:
            res = -xg1xg2 + x_g2g1
        elif i == 2:
            res = -xg1xg2 + x_g1g2
        elif i == 3:
            res = xg1xg2
        #         log2res = res.copy()
        #         log2res.data = np.log2(log2res.data)
        #         ents += -(res).multiply(log2res)
        log2res = res.copy()
        log2res.data[np.abs(log2res.data) < 1e-14] = 0.
        log2res.data = entr(log2res.data)
        ents += log2res / np.log(2)

    return ents, cov_norm


def calculate_independent_ents_sp(x_, xcov_, order=1):
    x = np.array(x_.todense())
    xcov = np.array(xcov_.todense())
    x[x>=0.5] = 1.
    x[x<0.5] = 0.
    x[xcov < 0.5] = np.nan
    probs = np.zeros(np.power(2, order))
    full = np.zeros([x.shape[1]] * order + [probs.shape[0]])
    cov_norm = np.zeros([x.shape[1]] * order)
    for i in range(probs.shape[0]):
        full[:,i] = 1. * np.nansum(x == i, axis=0) / np.nansum(~np.isnan(x), axis=0)
    full[np.abs(full) < 1e-14] = 0.
    return np.sum(entr(full), axis=-1) / np.log(2)

def calculate_VI(x_, xcov_, axis=0, order=2, idx=None, jdx=None, ignore_nans=True, probs=0):
    crossents, cov_norm = cross_entropy_by_batch_via_covarmatr(x_, xcov_, idx=idx, jdx=jdx)
    ind_ents = calculate_independent_ents_sp(x_, xcov_)
    denom = np.array(crossents.todense())
    crossents = np.array(crossents.todense())
    cov_norm = np.array(cov_norm.todense())
    denom[np.abs(denom) < 1e-3] = 1.
    VI = ((crossents - ind_ents.reshape(1,-1)) + (crossents - ind_ents.reshape(-1,1))) / denom
    VI = np.nan_to_num(VI)
    return VI, cov_norm

def MI_by_batch_via_covarmatr(x_, xcov_, axis=0, order=2, idx=None, jdx=None, ignore_nans=True):
    x = x_.copy()
    xcov = xcov_.copy()
    x[x >= 0.5] = 1.
    x[x < 0.5] = 0.
    if idx is None:
        idx = np.arange(x.shape[1])
    if jdx is None:
        jdx = np.arange(x.shape[1])
    Qg1g2 = sp.csr_matrix(sparse_cov_mat_by_batch(x, xcov, idx=idx, jdx=jdx))
    # x[xcov < 0.5] = 0.#np.nan
    xcov = 1. * (xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    x_mean = xcov.multiply(x).sum(axis=axis) / number_nonans
    x_g1 = sp.csr_matrix(np.broadcast_to(x_mean[:, idx].T, (len(idx), len(jdx))))
    x_g2 = sp.csr_matrix(np.broadcast_to(x_mean[:, jdx], (len(idx), len(jdx))))

    x_g1g2 = spa.tensordot((xcov[:, idx]).multiply(x[:, idx]), xcov[:, jdx], axes=([axis], [axis])).asformat(
        'csr').to_scipy_sparse()
    x_g2g1 = spa.tensordot((xcov[:, idx]), xcov[:, jdx].multiply(x[:, jdx]), axes=([axis], [axis])).asformat(
        'csr').to_scipy_sparse()
    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis])).asformat('csr').to_scipy_sparse()
    cov_norm.data = 1. / cov_norm.data

    x_g1g2 = x_g1g2.multiply(cov_norm)
    x_g2g1 = x_g2g1.multiply(cov_norm)

    xg1xg2 = (Qg1g2 +
              sp.csr_matrix(x_g1.multiply(x_g2g1)) +
              sp.csr_matrix(x_g2.multiply(x_g1g2)) -
              sp.csr_matrix(x_g1.multiply(x_g2)))
    ents = sp.lil_matrix(np.zeros((len(idx), len(jdx))))
    ents_1 = sp.lil_matrix(np.zeros((len(idx), len(jdx))))
    ents_2 = sp.lil_matrix(np.zeros((len(idx), len(jdx))))
    for i in range(2):
        if i == 0:
            res_1 = - x_g1g2
            res_1.data += 1.
            res_2 = - x_g2g1
            res_2.data += 1.
        elif i == 1:
            res_1 = x_g1g2
            res_2 = x_g2g1
        log2res_1 = res_1.copy()
        log2res_1.data[np.abs(log2res_1.data) < 1e-14] = 0.
        log2res_1.data = entr(log2res_1.data)
        ents_1 += log2res_1 / np.log(2)

        log2res_2 = res_2.copy()
        log2res_2.data[np.abs(log2res_2.data) < 1e-14] = 0.
        log2res_2.data = entr(log2res_2.data)
        ents_2 += log2res_2 / np.log(2)

    for i in range(4):
        if i == 0:
            res = xg1xg2 - x_g1g2 - x_g2g1
            res.data += 1.
        elif i == 1:
            res = -xg1xg2 + x_g2g1
        elif i == 2:
            res = -xg1xg2 + x_g1g2
        elif i == 3:
            res = xg1xg2
        #         log2res = res.copy()
        #         log2res.data = np.log2(log2res.data)
        #         ents += -(res).multiply(log2res)
        log2res = res.copy()
        log2res.data[np.abs(log2res.data) < 1e-14] = 0.
        log2res.data = entr(log2res.data)
        ents += log2res / np.log(2)

    denom = np.array(ents.todense()) * 0 + 1.
    # denom[np.abs(denom) < 1e-3] = 1.
    d_MI = ((2 * ents - ents_1 - ents_2)) / denom
    d_MI = np.nan_to_num(d_MI)
    d_MI = np.array(d_MI)
    return d_MI, np.array(cov_norm.todense())

def MI_distance_by_batch_via_covarmatr(x_, xcov_, axis=0, order=2, idx=None, jdx=None, ignore_nans=True):
    x = x_.copy()
    xcov = xcov_.copy()
    x[x >= 0.5] = 1.
    x[x < 0.5] = 0.
    if idx is None:
        idx = np.arange(x.shape[1])
    if jdx is None:
        jdx = np.arange(x.shape[1])
    Qg1g2 = sp.csr_matrix(sparse_cov_mat_by_batch(x, xcov, idx=idx, jdx=jdx))
    # x[xcov < 0.5] = 0.#np.nan
    xcov = 1. * (xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    x_mean = xcov.multiply(x).sum(axis=axis) / number_nonans
    x_g1 = sp.csr_matrix(np.broadcast_to(x_mean[:, idx].T, (len(idx), len(jdx))))
    x_g2 = sp.csr_matrix(np.broadcast_to(x_mean[:, jdx], (len(idx), len(jdx))))

    x_g1g2 = spa.tensordot((xcov[:, idx]).multiply(x[:, idx]), xcov[:, jdx], axes=([axis], [axis])).asformat(
        'csr').to_scipy_sparse()
    x_g2g1 = spa.tensordot((xcov[:, idx]), xcov[:, jdx].multiply(x[:, jdx]), axes=([axis], [axis])).asformat(
        'csr').to_scipy_sparse()
    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis])).asformat('csr').to_scipy_sparse()
    cov_norm.data = 1. / cov_norm.data

    x_g1g2 = x_g1g2.multiply(cov_norm)
    x_g2g1 = x_g2g1.multiply(cov_norm)

    xg1xg2 = (Qg1g2 +
              sp.csr_matrix(x_g1.multiply(x_g2g1)) +
              sp.csr_matrix(x_g2.multiply(x_g1g2)) -
              sp.csr_matrix(x_g1.multiply(x_g2)))
    ents = sp.lil_matrix(np.zeros((len(idx), len(jdx))))
    ents_1 = sp.lil_matrix(np.zeros((len(idx), len(jdx))))
    ents_2 = sp.lil_matrix(np.zeros((len(idx), len(jdx))))
    for i in range(2):
        if i == 0:
            res_1 = - x_g1g2
            res_1.data += 1.
            res_2 = - x_g2g1
            res_2.data += 1.
        elif i == 1:
            res_1 = x_g1g2
            res_2 = x_g2g1
        log2res_1 = res_1.copy()
        log2res_1.data[np.abs(log2res_1.data) < 1e-14] = 0.
        log2res_1.data = entr(log2res_1.data)
        ents_1 += log2res_1 / np.log(2)

        log2res_2 = res_2.copy()
        log2res_2.data[np.abs(log2res_2.data) < 1e-14] = 0.
        log2res_2.data = entr(log2res_2.data)
        ents_2 += log2res_2 / np.log(2)

    for i in range(4):
        if i == 0:
            res = xg1xg2 - x_g1g2 - x_g2g1
            res.data += 1.
        elif i == 1:
            res = -xg1xg2 + x_g2g1
        elif i == 2:
            res = -xg1xg2 + x_g1g2
        elif i == 3:
            res = xg1xg2
        #         log2res = res.copy()
        #         log2res.data = np.log2(log2res.data)
        #         ents += -(res).multiply(log2res)
        log2res = res.copy()
        log2res.data[np.abs(log2res.data) < 1e-14] = 0.
        log2res.data = entr(log2res.data)
        ents += log2res / np.log(2)

    denom = np.array(ents.todense())
    denom[np.abs(denom) < 1e-3] = 1.
    d_MI = ((2 * ents - ents_1 - ents_2)) / denom
    d_MI = np.nan_to_num(d_MI)
    d_MI = np.array(d_MI)
    return d_MI, np.array(cov_norm.todense())

def find_coreg_regions(ents, cov_norm, high_ent=0.2, low_ent=-100, coverage_coef=None, coverage_abs=None, cov_norm_th=None, num_of_NN=1.5):
    if cov_norm_th is None:

        if coverage_abs is None:
            if coverage_coef is None:
                coverage_coef = 1.
            mean_coverage = np.nanmean(1. / cov_norm[cov_norm > 0.])
            # mean_coverage = np.nanmean(np.nansum(1. * (cov_norm > 1e-5), axis=0))
            coverage_threshold = coverage_coef * mean_coverage
        else:
            coverage_threshold = coverage_abs
        coverage_threshold = np.zeros_like(cov_norm) + coverage_threshold
    else:
        coverage_threshold = cov_norm_th

    jdx_coreg = np.zeros(ents.shape[0]) == 1
    jdx_stoch = np.zeros(ents.shape[0]) == 1

    for i in range(ents.shape[0]):
        idx = np.logical_and(ents[i, :] > low_ent, ents[i, :] < high_ent)
        idx = np.logical_and(np.logical_and(idx, 1. / cov_norm[i, :] >= coverage_threshold[i,:]), cov_norm[i, :] > 1e-5)
        if (np.sum(1. * idx) > num_of_NN):
            jdx_coreg[idx] = True
        elif (np.sum(1. * idx) < num_of_NN):
            jdx_stoch[idx] = True

    jdx_lowcov = np.logical_not(np.logical_or(jdx_coreg, jdx_stoch))
    return jdx_coreg, jdx_stoch, jdx_lowcov


def find_coreg_regions_smart(ents, cov_norm, high_ent=0.2, low_ent=-100, coverage=1, cluster_size=2.):
    coverage_threshold = np.zeros_like(ents) + 1
    coverage_threshold[cov_norm > 1e-5] = np.power(2, - (1. / cov_norm[cov_norm > 1e-5] - 1))
    thres = np.sum(coverage_threshold, axis=1) + 1.

    jdx_lowcov = np.sum(1. * np.logical_and(1. / cov_norm >= coverage, cov_norm > 1e-5), axis=1) < 1.5
    idx = np.sum(np.logical_and(ents > low_ent, ents < high_ent), axis=1) > thres + cluster_size
    jdx_coreg = np.logical_and(idx, np.logical_not(jdx_lowcov))
    jdx_stoch = np.logical_and(np.logical_not(idx), np.logical_not(jdx_lowcov))
    return jdx_coreg, jdx_stoch, jdx_lowcov


from scipy.interpolate import splev, splrep


def generate_fully_random_samples(Nsam=5, Ng=100):
    x = np.random.rand(Nsam, Ng)
    xcov = np.random.randn(Nsam, Ng) * 0 + 1.

    x_sp_corr = sp.csr_matrix(np.nan_to_num(x))
    xt_sp_corr = sp.csr_matrix(np.nan_to_num(x.T))
    xcov_sp_corr = sp.csr_matrix(np.nan_to_num(xcov))
    return x, xcov, x_sp_corr, xcov_sp_corr


def get_d_MI(l=5, Ng=100):
    x, xcov, x_sp, xcov_sp = generate_fully_random_samples(Nsam=l, Ng=Ng)

    # r_fast, r_cov_norm_fast = anatools.sparse_corr_mat_by_batch(x_sp, xcov_sp)
    d_MI, cov_norm_MI = MI_by_batch_via_covarmatr(x_sp, xcov_sp, order=2)
    d_MI = d_MI[np.eye(d_MI.shape[0]) != 1].flatten()
    # r_fast = r_fast[np.eye(r_fast.shape[0]) != 1].flatten()
    return d_MI


def get_threshold(d_MI, d0=0.0):
    return 1. * np.sum(d_MI <= d0) / np.prod(d_MI.shape)

def get_G_th(N_max, d0=0.):
    G_th = np.zeros(N_max) + 1.
    for l in range(1, N_max):
        if l > 30:
            G_th[l - 1] = np.power(2., -l + 1)
        else:
            G_th[l - 1] = get_threshold(get_d_MI(l=l, Ng=100), d0 = d0 + np.power(2., -l))
            if G_th[l - 1] < np.power(2., -l + 1):
                G_th[l - 1] = np.power(2., -l + 1)
    x = np.arange(1, G_th.shape[0] + 1)
    spl = splrep(x, G_th, s=0, k =1)
    return lambda x: np.clip(splev(x, spl), 0, 1)
    # return G_th
    #return lambda x: splev(x, spl)

def get_G_th_fast(N_max, d0=0.):
    G_th = np.zeros(N_max) + 1.
    for l in range(1, N_max + 1):
        if l > 30:
            G_th[l - 1] = np.power(2., -l + 1)
        else:
            G_th[l - 1] = get_threshold(get_d_MI(l=l, Ng=100), d0 = d0 + np.power(2., -l))
            if G_th[l - 1] < np.power(2., -l + 1):
                G_th[l - 1] = np.power(2., -l + 1)
    G_th[1:] = G_th[:-1]
    return np.clip(G_th, 0, 1)

def find_coreg_regions_smart_adaptive(ents, cov_norm, high_ent=0.2, low_ent=-100, coverage=1, cluster_size=2., calc_G_th=None):
    coverage_threshold = np.zeros_like(ents) + 1
    if calc_G_th is None:
        calc_G_th = get_G_th(int(np.max(1./(cov_norm[cov_norm > 1e-5]))), high_ent)
    coverage_threshold[cov_norm > 1e-5] = calc_G_th(1. / cov_norm[cov_norm > 1e-5])
    thres = np.sum(coverage_threshold, axis=1) + 1.

    jdx_lowcov = np.sum(1. * np.logical_and(1. / cov_norm >= coverage, cov_norm > 1e-5), axis=1) < 1.5
    idx = np.sum(np.logical_and(ents > low_ent, ents < high_ent), axis=1) > thres + cluster_size
    jdx_coreg = np.logical_and(idx, np.logical_not(jdx_lowcov))
    jdx_stoch = np.logical_and(np.logical_not(idx), np.logical_not(jdx_lowcov))
    return jdx_coreg, jdx_stoch, jdx_lowcov

def cov2int(x):
    x = np.array(x, dtype=np.float16)
    y = np.zeros_like(x, dtype=np.int32)
    y[x > 1e-5] = np.array(1./x[x > 1e-5], dtype=np.int32)
    return y

def find_coreg_regions_smart_adaptive_fast(ents, cov_norm, high_ent=0.2, low_ent=-100, coverage=1, cluster_size=2., calc_G_th=None, idx_chosen=None):
    if idx_chosen is None:
        idx_chosen = np.zeros(ents.shape[1]) == 0
    if calc_G_th is None:
        calc_G_th = get_G_th(np.max(cov_norm), high_ent)
    coverage_threshold = calc_G_th[cov_norm]
    thres = np.sum(coverage_threshold, axis=1) + 1.

    jdx_lowcov = np.sum(cov_norm >= coverage, axis=1) < 1.5
    idx = np.sum(ents < high_ent, axis=1) > (thres + cluster_size)
    jdx_coreg = np.logical_and(idx, np.logical_not(jdx_lowcov))
    jdx_stoch = np.logical_and(np.logical_not(idx), np.logical_not(jdx_lowcov))
    return jdx_coreg, jdx_stoch, jdx_lowcov

def find_coreg_in_hdf5_fast(filename, coverage=4, cluster_size=3, high_ent=0.75, N_max=300, nbatch=None, idx_chosen=None):
    hf = h5py.File(filename, 'r')
    keys = hf.keys()
    if nbatch is None:
        nbatch = np.max([int(x.split('_')[1]) for x in keys])
    batch = hf.get(f'data_{0}').shape[0]
    G = hf.get(f'data_{0}').shape[1]
    if G > N_max:
        N_max = G
    calc_G_th = get_G_th_fast(N_max, high_ent)
    idx_coreg, idx_stoch, idx_lowq = np.zeros(G) == 1, np.zeros(G) == 1, np.zeros(G) == 1
    for i in trange(nbatch):
        idx = np.arange(i * batch, np.min([ (i + 1) * batch, G]))
        idx_coreg[idx], idx_stoch[idx], idx_lowq[idx] = find_coreg_regions_smart_adaptive_fast(np.array(hf.get(f'data_{i}'), dtype=np.float16), cov2int(hf.get(f'cov_{i}')),
                                                                                    coverage=coverage, cluster_size=cluster_size, high_ent=high_ent, calc_G_th=calc_G_th,
                                                                                    idx_chosen=idx_chosen
                                                                                   )
    return idx_coreg, idx_stoch, idx_lowq

def find_coreg_in_hdf5(filename, coverage=4, cluster_size=3, high_ent=0.75, N_max=50, nbatch=None):
    hf = h5py.File(filename, 'r')
    keys = hf.keys()
    if nbatch is None:
        nbatch = np.max([int(x.split('_')[1]) for x in keys])
    calc_G_th = get_G_th(N_max, high_ent)
    batch = hf.get(f'data_{0}').shape[0]
    G = hf.get(f'data_{0}').shape[1]
    idx_coreg, idx_stoch, idx_lowq = np.zeros(G) == 1, np.zeros(G) == 1, np.zeros(G) == 1
    for i in trange(nbatch):
        idx = np.arange(i * batch, np.min([ (i + 1) * batch, G]))
        idx_coreg[idx], idx_stoch[idx], idx_lowq[idx] = find_coreg_regions_smart_adaptive(np.array(hf.get(f'data_{i}'), dtype=np.float16), np.array(hf.get(f'cov_{i}'), dtype=np.float16),
                                                                                    coverage=coverage, cluster_size=cluster_size, high_ent=high_ent, calc_G_th=calc_G_th
                                                                                   )
    return idx_coreg, idx_stoch, idx_lowq

def find_coreg_regions_prob(ents, cov_norm, high_ent=0.2, low_ent=-100, coverage=1, cluster_size=2.):
    coverage_threshold = np.zeros_like(ents) + 1
    coverage_threshold[cov_norm > 1e-5] = np.power(2, - (1. / cov_norm[cov_norm > 1e-5] - 1))
    thres = np.sum(coverage_threshold, axis=1) + 1.

    jdx_lowcov = np.sum(1. * np.logical_and(1. / cov_norm >= coverage, cov_norm > 1e-5), axis=1) < 1.5
    idx = np.sum(np.logical_and(ents > low_ent, ents < high_ent), axis=1) > thres + cluster_size
    jdx_coreg = np.logical_and(idx, np.logical_not(jdx_lowcov))
    jdx_stoch = np.logical_and(np.logical_not(idx), np.logical_not(jdx_lowcov))
    return jdx_coreg, jdx_stoch, jdx_lowcov

def find_coreg_regions_no_coverage(ents, cov_norm, high_ent=0.2, low_ent=-100, coverage_coef=None, coverage_abs=None, cov_norm_th=None, num_of_NN=1.5):
    jdx_coreg = np.zeros(ents.shape[0]) == 1
    jdx_stoch = np.zeros(ents.shape[0]) == 1

    for i in range(ents.shape[0]):
        idx = np.logical_and(ents[i, :] > low_ent, ents[i, :] < high_ent)
        if (np.sum(idx) > num_of_NN):
            jdx_coreg[idx] = True
        elif (np.sum(idx) < num_of_NN):
            jdx_stoch[idx] = True

    jdx_lowcov = np.logical_not(np.logical_or(jdx_coreg, jdx_stoch))
    return jdx_coreg, jdx_stoch, jdx_lowcov

def mask_0_1_tonan(x):
    if isinstance(x, np.ndarray):
        if np.nanmax(x) > 1.1:
            x /= 100.
        x[x <= 1e-6] = np.nan
        x[x >= 1. - 1e-6] = np.nan
    else:
        # if np.max(x.data) > 1.1:
        #     x.data /= 100.
        pass
    return x

def logit(x):
    if isinstance(x, np.ndarray):
        return scipy.special.logit(x)
    else:
        y = x.copy()
        y.data = scipy.special.logit(y.data)
        return y


def cov_mat_by_batch_fast(x_, xcov_, axis=0, order=2, idx=None, jdx=None, ignore_nans=True):
    x = x_.copy()
    xcov = xcov_.copy()
    x[x >= 0.5] = 1.
    x[x < 0.5] = 0.
    if idx is None:
        idx = np.arange(x.shape[1])
    if jdx is None:
        jdx = np.arange(x.shape[1])
    xcov = 1. * (xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    x_mean = (xcov * x).sum(axis=axis).reshape(1, -1) / number_nonans

    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis]))
    cov = (spa.tensordot(x[:, idx] * xcov[:, idx], x[:, jdx] * xcov[:, jdx], axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx] * x_mean[:, idx], xcov[:, jdx] * x[:, jdx],
                         axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx] * x[:, idx], xcov[:, jdx] * x_mean[:, jdx],
                         axes=([axis], [axis])) +
           spa.tensordot(xcov[:, idx] * x_mean[:, idx], xcov[:, jdx] * x_mean[:, jdx],
                         axes=([axis], [axis])))
    # cov = np.array(cov.todense())
    # cov_norm = np.array(cov_norm.todense())
    return cov / cov_norm

def MI_by_batch_fast(x_, xcov_, axis=0, order=2, idx=None, jdx=None, ignore_nans=True):
    x = x_.copy()
    xcov = xcov_.copy()
    x[x >= 0.5] = 1.
    x[x < 0.5] = 0.
    if idx is None:
        idx = np.arange(x.shape[1])
    if jdx is None:
        jdx = np.arange(x.shape[1])
    # Qg1g2 = sparse_cov_mat_by_batch_TEST(x, xcov, idx=idx, jdx=jdx)
    # x[xcov < 0.5] = 0.#np.nan
    # x = np.array(x.todense())
    # xcov = np.array(xcov.todense())

    xcov = 1. * (xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    x_mean = (xcov * x).sum(axis=axis).reshape(1, -1) / number_nonans

    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis]))
    cov = (spa.tensordot(x[:, idx] * xcov[:, idx], x[:, jdx] * xcov[:, jdx], axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx] * x_mean[:, idx], xcov[:, jdx] * x[:, jdx],
                         axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx] * x[:, idx], xcov[:, jdx] * x_mean[:, jdx],
                         axes=([axis], [axis])) +
           spa.tensordot(xcov[:, idx] * x_mean[:, idx], xcov[:, jdx] * x_mean[:, jdx],
                         axes=([axis], [axis])))
    # cov = np.array(cov.todense())
    # cov_norm = np.array(cov_norm.todense())
    Qg1g2 = cov / cov_norm

    x_g1 = np.broadcast_to(x_mean[:, idx].T, (len(idx), len(jdx)))
    x_g2 = np.broadcast_to(x_mean[:, jdx], (len(idx), len(jdx)))

    x_g1g2 = spa.tensordot(xcov[:, idx] * x[:, idx], xcov[:, jdx],
                           axes=([axis], [axis]))  # .asformat('csr').to_scipy_sparse()
    x_g2g1 = spa.tensordot(xcov[:, idx], xcov[:, jdx] * x[:, jdx],
                           axes=([axis], [axis]))  # .asformat('csr').to_scipy_sparse()
    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis]))  # .asformat('csr').to_scipy_sparse()
    cov_norm = 1. / cov_norm
    x_g1g2 = x_g1g2 * cov_norm
    x_g2g1 = x_g2g1 * cov_norm
    # x_g1g2 = x_g1g2.multiply(cov_norm)
    # x_g2g1 = x_g2g1.multiply(cov_norm)
    xg1xg2 = (Qg1g2 +
              x_g1 * x_g2g1 +
              x_g2 * x_g1g2 -
              x_g1 * x_g2)
    ents = np.zeros((len(idx), len(jdx)))
    ents_1 = np.zeros((len(idx), len(jdx)))
    ents_2 = np.zeros((len(idx), len(jdx)))
    for i in range(2):
        if i == 0:
            res_1 = - x_g1g2
            res_1 += 1.
            res_2 = - x_g2g1
            res_2 += 1.
        elif i == 1:
            res_1 = x_g1g2
            res_2 = x_g2g1
        log2res_1 = res_1.copy()
        log2res_1[np.abs(log2res_1) < 1e-14] = 0.
        log2res_1 = entr(log2res_1)
        ents_1 += log2res_1 / np.log(2)

        log2res_2 = res_2.copy()
        log2res_2[np.abs(log2res_2) < 1e-14] = 0.
        log2res_2 = entr(log2res_2)
        ents_2 += log2res_2 / np.log(2)

    for i in range(4):
        if i == 0:
            res = xg1xg2 - x_g1g2 - x_g2g1
            res += 1.
        elif i == 1:
            res = -xg1xg2 + x_g2g1
        elif i == 2:
            res = -xg1xg2 + x_g1g2
        elif i == 3:
            res = xg1xg2
        #         log2res = res.copy()
        #         log2res.data = np.log2(log2res.data)
        #         ents += -(res).multiply(log2res)
        log2res = res.copy()
        log2res[np.abs(log2res) < 1e-14] = 0.
        log2res = entr(log2res)
        ents += log2res / np.log(2)

    denom = ents * 0 + 1.
    # denom[np.abs(denom) < 1e-3] = 1.
    d_MI = ((2 * ents - ents_1 - ents_2)) / denom
    d_MI = np.nan_to_num(d_MI)
    cov_norm[cov_norm > 100] = 0.
    return d_MI, cov_norm


def corr_mat_by_batch_fast(x_, xcov_, axis=0, order=2, idx=None, jdx=None, ignore_nans=True):
    x = x_.copy()
    xcov = xcov_.copy()
    if idx is None:
        idx = np.arange(x.shape[1])
    if jdx is None:
        jdx = np.arange(x.shape[1])
    xcov = 1. * (xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    x_mean = (xcov * x).sum(axis=axis).reshape(1, -1) / number_nonans

    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis]))
    cov = (spa.tensordot(x[:, idx] * xcov[:, idx], x[:, jdx] * xcov[:, jdx], axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx] * x_mean[:, idx], xcov[:, jdx] * x[:, jdx],
                         axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx] * x[:, idx], xcov[:, jdx] * x_mean[:, jdx],
                         axes=([axis], [axis])) +
           spa.tensordot(xcov[:, idx] * x_mean[:, idx], xcov[:, jdx] * x_mean[:, jdx],
                         axes=([axis], [axis])))
    # cov = np.array(cov.todense())
    # cov_norm = np.array(cov_norm.todense())

    cov_denomin = ((xcov * x * x).sum(axis=axis) -
                   (xcov * np.power(x_mean, 2)).sum(axis=axis)).reshape(1, -1)
    cov_denomin = 1. / np.sqrt(cov_denomin)
    cov_mat = cov * cov_denomin[:, jdx]
    cov_mat = cov_denomin[:, idx].T * cov_mat
    cov_norm = 1. / cov_norm
    cov_norm[cov_norm > 100] = 0.
    return cov_mat, cov_norm



def corr_mcc_mat_by_batch_fast(x_, xcov_, axis=0, order=2, idx=None, jdx=None, ignore_nans=True):
    x = x_.copy()
    xcov = xcov_.copy()
    if idx is None:
        idx = np.arange(x.shape[1])
    if jdx is None:
        jdx = np.arange(x.shape[1])
    xcov = 1. * (xcov > 0.5)
    number_nonans = xcov.sum(axis=axis)
    number_nonans[np.abs(number_nonans) < 0.1] = np.nan
    x_mean = (xcov * x).sum(axis=axis).reshape(1, -1) / number_nonans

    cov_norm = spa.tensordot(xcov[:, idx], xcov[:, jdx], axes=([axis], [axis]))
    cov = (spa.tensordot(x[:, idx] * xcov[:, idx], x[:, jdx] * xcov[:, jdx], axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx] * x_mean[:, idx], xcov[:, jdx] * x[:, jdx],
                         axes=([axis], [axis])) -
           spa.tensordot(xcov[:, idx] * x[:, idx], xcov[:, jdx] * x_mean[:, jdx],
                         axes=([axis], [axis])) +
           spa.tensordot(xcov[:, idx] * x_mean[:, idx], xcov[:, jdx] * x_mean[:, jdx],
                         axes=([axis], [axis])))
    # cov = np.array(cov.todense())
    # cov_norm = np.array(cov_norm.todense())

    cov_denomin = ((xcov * x * x).sum(axis=axis) -
                   (xcov * np.power(x_mean, 2)).sum(axis=axis)).reshape(1, -1)
    cov_denomin = 1. / np.sqrt(cov_denomin)
    cov_mat = cov * cov_denomin[:, jdx]
    cov_mat = cov_denomin[:, idx].T * cov_mat
    cov_norm = 1. / cov_norm
    cov_norm[cov_norm > 100] = 0.
    return cov_mat, cov_norm