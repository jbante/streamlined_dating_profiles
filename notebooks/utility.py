# common utility functions for the analysis
import itertools as itr
import time
from datetime import timedelta as td

import joblib as jl
import matplotlib.pyplot as plt
import multiprocess as mp
import numba as nb
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.spatial.distance as dist
import scipy.stats as st

DEFAULT_RNG = np.random.default_rng()


def add_prefix_to_columns(df, prefix):
    """Adds a prefix to the columns of a DataFrame"""
    df.columns = [prefix + col for col in df.columns]
    return df


def amissing(x, missing_ind):
    """Boolean array marking missing values"""
    return np.isnan(x) if np.isnan(missing_ind) else x == missing_ind


def bits_estimate(n):
    """
    Estimated bits needed to describe every individual in a dataset of a given size
    """
    return iceil(np.log2(n))


def counts_within_ball(dist_matrix, epsilon):
    """Number of points in a closed epsilon-ball around each point"""
    return np.sum(dist_matrix <= epsilon, axis=0)


def dist_matrix_and_dim(X, metric):
    """
    Returns distance and dimension information

    Parameters
    ----------
    X : array
    metric : function

    Returns
    -------
    dist_matrix : array
        distance matrix
    implied_dim : float
    max_dim : int
    """
    # precompute distance matrix to accelerate parameter search
    dists = pdist(X.values, metric, n_jobs=-1)
    dist_matrix = dist.squareform(dists)

    # estimate maximum reasonable dimensions
    implied_dim = implied_dimension_from_matrix(dist_matrix, dists)
    max_dim = min(iceil(implied_dim), max_entropy_dimension(X.shape[0]))
    return dist_matrix, implied_dim, max_dim


@nb.jit
def dist_matrix_split(dist_matrix, subset):
    """Split of a distance matrix specified by a boolean list"""
    return (
        dist_matrix_subset(dist_matrix, subset),
        dist_matrix_subset(dist_matrix, ~subset),
    )


@nb.jit
def dist_matrix_subset(dist_matrix, subset):
    """Returns a subset of a distance matrix specified by a boolean list"""
    return dist_matrix[subset].T[subset]


def extrema_indices(X):
    """Returns the indices for the extrema of vectors in X along each coordinate axis"""
    return list(set((flat_list([[np.argmin(a), np.argmax(a)] for a in X.T]))))


def flat_list(list_of_lists):
    """flattens a list of lists into a single list"""
    return np.array(list(itr.chain(*list_of_lists)))


def iceil(x):
    """Integer-type ceiling"""
    return np.ceil(x).astype(int)


def ifloor(x):
    """Integer-type floor"""
    return np.floor(x).astype(int)


def implied_dimension(data, metric, n_samples=1000, rng=DEFAULT_RNG):
    """Spatial dimension implied by data and a metric over that data"""
    x = np.asanyarray(data)
    n_samples = min(n_samples, len(x))
    if n_samples < len(x):
        sample_indices = rng.choice(len(x), n_samples, replace=False)
        x = x[sample_indices]

    dist_list = pdist(x, metric, n_jobs=-1)
    dist_matrix = dist.squareform(dist_list)
    return _implied_dimension_from_matrix(dist_matrix, dist_list)


def implied_dimension_from_matrix(dist_matrix, dist_list):
    """Spatial dimension implied by a distance matrix"""
    # count how many points are contained in epsilon-balls centered on points
    n = len(dist_list)
    n_samples = iceil(np.sqrt(n))
    sample_indices = np.linspace(0, n / 2, n_samples).astype(int)
    radii = np.sort(dist_list)[sample_indices]
    x = []
    y = []
    for epsilon in radii:
        points_in_balls = counts_within_ball(dist_matrix, epsilon)
        x.extend([epsilon] * len(points_in_balls))
        y.extend(points_in_balls)

    # calculate dimension: y = c * x**dim -> log(y) = dim * log(x) + log(c)
    A = np.vstack([np.log(x), np.ones(len(x))]).T

    try:
        dim = la.lstsq(A, np.log(y))[0][0]
    except:
        dim = bits_estimate(len(dist_matrix))

    return dim


def indicate_missing_data(df, missing, indicator=np.nan):
    """Sets an indicator value for rows in df that should be marked as missing"""
    for col in df.columns:
        df[col] = [indicator if m else x for x, m in zip(df[col], missing)]

    return df


def initial_medoids_extrema(X, n_medoids):
    """
    Returns indices to initialize k-Medoids clustering by selecting the dimension-wise
    extrema
    """
    return initial_medoids_random(len(X), n_medoids, extrema_indices(X))


def initial_medoids_random(n_data, n_medoids, seed_indices=None, rng=DEFAULT_RNG):
    """Returns random indices to initialize k-medoids clustering"""
    if seed_indices is not None and len(seed_indices) >= n_medoids:
        return seed_indices[:n_medoids]

    n_range = range(n_data)
    if n_data <= n_medoids:
        return list(n_range)

    indices = seed_indices if seed_indices is not None else []
    remaining = n_medoids - len(indices)
    if remaining > 0:
        options = [i for i in n_range if i not in indices]
        indices.extend(rng.choice(options, size=remaining, replace=False))

    return indices


@nb.jit
def invert_index_lists(list_of_lists):
    """Converts a list of lists of indices into a flat list of class labels"""
    n_labels = len(list_of_lists)
    n = np.sum([len(sub) for sub in list_of_lists])
    result = np.zeros(n)
    if n_labels < 2:
        return result

    # first class is already in result as 0, so we can skip it here
    for label, sublist in zip(range(1, n_labels), list_of_lists[1:]):
        for i in sublist:
            result[i] = label

    return result.astype(int)


def is_sharp_vertex(points, x):
    """Returns True if all points lie within an orthogonal cone with x at the apex"""
    return all([np.dot(a - x, b - x) > 0 for a, b in itr.combinations(points, 2)])


def match_any_prefix(text, prefixes):
    """True if text begins with any value in prefixes"""
    return any([text.startswith(pre) for pre in prefixes])


def match_bits_estimate(n):
    """
    Estimated number of bits needed to characterize every pairwise match in a dataset
    of a given size
    """
    return basic_bits_estimate((n - 1) * (n - 2) / 2)


def max_entropy_dimension(n):
    """Maximum number of dimensions that could plausibly be supported by n points"""
    return ifloor(np.log2(n))


def mean_count_within_ball(dist_matrix, epsilon):
    """Mean number of points within an epsilon-ball around each point"""
    return np.mean(dist_matrix <= epsilon)


def multi_dummies(series, col_prefix, delimiter=', ', missing_ind=np.nan):
    """Expands a multivalue column into dummies"""
    dummies = series.str.lower().str.get_dummies(delimiter)
    unknowns = series.apply(value_is_nan)
    indicate_missing_data(dummies, unknowns, missing_ind)
    return add_prefix_to_columns(dummies, col_prefix)


def nan_matrix_indices(dist_matrix):
    """Indices of any np.nan values in dist_matrix"""
    n_rows, n_cols = dist_matrix.shape
    return [
        (i, j)
        for i, j in itr.product(range(n_rows), range(n_cols))
        if (i < j) and np.isnan(dist_matrix[i, j])
    ]


def neg_matrix_indices(dist_matrix):
    """Indices of any negative values in dist_matrix"""
    n_rows, n_cols = dist_matrix.shape
    return [
        (i, j)
        for i, j in itr.product(range(n_rows), range(n_cols))
        if (i < j) and (dist_matrix[i, j] < 0)
    ]


def non_missing_series(series, missing_ind):
    """A series with any missing values removed"""
    if np.isnan(missing_ind):
        return series.dropna()
    else:
        return series[series != missing_ind]


def pair_is_binary(pair):
    return len(pair) == 2 and pair[0] == 0 and pair[1] == 1


def pdist(data, dist_func, n_jobs=1, max_time=None):
    """
    Parallel pairwise distances between elements of x. Similar to scipy pdist with
    parallel processing and bounded runtime.
    """
    x = np.asanyarray(data)
    n_jobs = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
    if max_time is None or max_time == np.inf:
        with mp.Pool(processes=n_jobs) as pool:
            dists = pool.map(lambda a: dist_func(*a), itr.combinations(x, 2))
        return np.array(dists)

    def dist_col(a, b_list):
        return [dist_func(a, b) for b in b_list]

    dist_lists = []
    end = np.inf if max_time is None else time.time() + max_time
    projected_end = np.NINF
    n = len(x)
    i = 0
    end_i = 0

    # calculate matrix columns in chunks of n_jobs at a time
    with jl.Parallel(n_jobs=n_jobs) as parallel:
        while end_i < n and projected_end <= end:
            chunk_start = time.time()
            start_i = i + 1
            end_i = min(i + n_jobs + 1, n)
            a_list = [x[j] for j in range(start_i, end_i)]
            b_lists = [x[:k] for k in range(start_i, end_i)]

            cols = parallel(jl.delayed(dist_col)(a, b) for a, b in zip(a_list, b_lists))

            for col in cols:
                for sublist, d in zip(dist_lists, col):
                    sublist.append(d)
                dist_lists.append([col[-1]])

            i += n_jobs
            n_items = sum([len(b) for b in b_lists])
            next_items = n_items + n_jobs * (n_jobs - 1) / 2
            col_time = time.time() - chunk_start
            projected_end = time.time() + col_time * next_items / n_items

    return flat_list(dist_lists)


def runtimer(title=None):
    """
    Function that prints the elapsed time since it was created (for speed testing)
    """
    start = time.time()

    def f():
        runtime = td(seconds=time.time() - start)
        if title is None:
            print(f'runtime: {runtime}')
        else:
            print(f'{title} runtime: {runtime}')
        return None

    return f


def sharp_vertices_indices(X):
    """Returns the indices for any 'sharp' convex vertices in a set of points"""
    indices = np.arange(len(X))
    return [i for i in indices if is_sharp_vertex(X[indices != i], X[i])]


def value_is_nan(x):
    """Returns true if x is NaN; safe for x of unknown type"""
    return isinstance(x, float) and np.isnan(x)
