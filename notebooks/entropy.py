# proof of concept entropy calculations accounting for missing values
# implementations only support nominal data
import functools as fn
import itertools as itr

import numba as nb
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

########################################################################################
# private entropy & utility functions
########################################################################################


@nb.jit
def _column_sum(x):
    """
    Sum of values in each column (since Numba doesn't support np.sum using axis
    argument)
    """
    y = x.T
    n = len(y)
    result = np.zeros(n)
    for i in range(n):
        result[i] = np.sum(y[i])
    return result


@nb.jit
def _conditional_entropy(cont_prob_table):
    """Entropy of the first axis variable conditional on the second axis variable"""
    weights = _column_sum(cont_prob_table)
    table = cont_prob_table.T
    n = len(table)
    h = 0
    for i in range(n - 1):
        h += weights[i] * _entropy_from_sizes(table[i])
    unconditional = _column_sum(table)
    h += _entropy_from_sizes(unconditional) * weights[-1]
    return h


@nb.jit
def _contingency_prob_table(a, b, missing_ind):
    """
    Contingency probability table for two paired arrays; last values are for missing
    values
    """
    a_values = _unique(a, missing_ind)
    b_values = _unique(b, missing_ind)
    a_n = len(a_values)
    b_n = len(b_values)

    # verbose looping necessary for Numba, which has limited support for np.sum
    ab_counts = np.empty(a_n * b_n)
    i = 0
    for a_val in a_values:
        for b_val in b_values:
            ab_counts[i] = np.sum(_eq_nan(a, a_val) & _eq_nan(b, b_val))
            i += 1

    ab_counts = np.reshape(ab_counts, (a_n, b_n))
    ab_probs = ab_counts / len(a)
    return ab_probs


def _either_mask_func(mask_func):
    """Returns a function that returns array OR combination of a mask on two inputs"""

    @nb.jit
    def either(a, b):
        return mask_func(a) | mask_func(b)

    return either


@nb.jit
def _entropy(x, missing_ind=np.nan):
    """Entropy of categorical data in x, accounting for missing values"""
    return _entropy_from_sizes(_unique_counts(x, missing_ind))


@nb.jit
def _entropy_from_sizes(x):
    """
    Entropy based on an array containing class sizes, assuming the last entry is for
    missing values
    """
    from_n = np.sum(x)
    # with only 1 non-missing unique value, there is no entropy
    if len(x) <= 2 or from_n == 0:
        return 0

    from_size = x / from_n
    to_n = from_n - x[-1]
    to_size = np.array([to_n / a if a != 0 else 1 for a in x])
    to_size[-1] = 1
    return np.sum(from_size * np.log2(to_size))


@nb.jit
def _eq_nan(a, b):
    """True if the arguments are equal, so that np.nan == np.nan"""
    return (np.isnan(a) & np.isnan(b)) | (a == b)


def _missing_func(missing_ind):
    """Returns a function that returns True if it's argument is a missing value"""
    if np.isnan(missing_ind):

        @nb.jit
        def f(x):
            return np.isnan(x)

    else:

        @nb.jit
        def f(x):
            return x == missing_ind

    return f


@nb.jit
def _missing_mask(x, missing_ind):
    """Boolean array marking locations of missing values"""
    return np.isnan(x) if np.isnan(missing_ind) else x == missing_ind


def _missing_mask_func(missing_ind):
    """Returns function for boolean array marking locations of missing values"""
    if np.isnan(missing_ind):

        @nb.jit
        def mask(x):
            return np.isnan(x)

    else:

        @nb.jit
        def mask(x):
            return x == missing_ind

    return mask


def _ndarray_from_mixed_list(mixed_lists):
    """Converts a list of mixed-length lists into a uniform numpy array"""
    n_rows = len(mixed_lists)
    n_cols = max([len(sublist) for sublist in mixed_lists])
    result = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j, value in enumerate(mixed_lists[i]):
            result[i, j] = mixed_lists[i][j]
    return result


def _neither_is_missing_func(missing_ind):
    """Returns a function that indicates whether two values are both not missing"""
    if np.isnan(missing_ind):

        @nb.jit
        def f(a, b):
            return not (np.isnan(a) or np.isnan(b))

    else:

        @nb.jit
        def f(a, b):
            return not (a == missing_ind or b == missing_ind)

    return f


@nb.jit
def _normalized_entropy(x, missing_ind=np.nan):
    """
    Entropy of data in x, normalized relative to perfect entropy for the cardinality
    """
    counts = _unique_counts(x, missing_ind)
    n_categories = len(counts) - 1
    if n_categories <= 1:
        return 0
    else:
        return _entropy_from_sizes(counts) / np.log2(n_categories)


def _shape_type_guard(func):
    """
    Creates a wrapper function that guards for early-return conditions and handles
    multiple columns and Pandas types
    """

    def f(x, **kwargs):
        if len(x) == 0:
            return 0
        elif type(x) == pd.core.series.Series:
            return func(x.values, **kwargs)
        elif type(x) == pd.core.frame.DataFrame:
            return pd.Series(f(x.values, **kwargs), x.columns)

        x = np.asanyarray(x)
        if len(x.shape) > 1:
            return np.array([func(col, **kwargs) for col in x.T])
        else:
            return func(x, **kwargs)

    return f


@nb.jit
def _unique_counts(x, missing_ind):
    """Counts of unique values in an array; not necessarily aligned with _unique"""
    missing = _missing_mask(x, missing_ind)
    y = [int(a) for a in x[~missing]]
    return np.append(np.bincount(y), np.sum(missing))


@nb.jit
def _unique(x, missing_ind):
    """Unique values in an array, without duplicating missing indicator like np.nan"""
    values = np.unique(x)
    return np.append(values[~_missing_mask(values, missing_ind)], missing_ind)


@nb.jit
def _vi(a, b, missing_ind):
    """Variation of information (distance) between a and b"""
    ab = _contingency_prob_table(a, b, missing_ind)
    return _conditional_entropy(ab) + _conditional_entropy(ab.T)


########################################################################################
# private hamming distance functions
########################################################################################


def _hamming_dist_entropy_func(data, missing_ind):
    """
    Returns a function that calculates a variation of Hamming distance that interprets
    Hamming distance as a measure of set size, assuming that features are independently
    distributed and no missing values will be compared.

    Parameters
    ----------
    data : array_like
        Distances will be calculated with respect to the distribution of values in data
    missing_ind
        The indicator used in data to represent missing values
    """
    n_data = data.shape[0]
    log_n_data = np.log2(n_data)
    n_features = data.shape[1]
    counts = []
    nan_counts = np.zeros(n_features)
    for i, feature in enumerate(data.T):
        u_counts = _unique_counts(feature, missing_ind)
        counts.append(u_counts)
        nan_counts[i] = n_data - np.sum(u_counts)

    value_counts = _ndarray_from_mixed_list(counts)

    @nb.jit
    def dist(a, b, i):
        a_size = value_counts[i][int(a)]
        b_size = value_counts[i][int(b)]
        size = a_size if a == b else a_size + b_size
        adjusted_size = size + nan_counts[i]
        return np.log2(adjusted_size)

    @nb.jit
    def dists_func(a, b, not_missing):
        itr_not_missing = enumerate(zip(not_missing, a, b))
        dists = np.array(
            [dist(a_val, b_val, i) for i, (nm, a_val, b_val) in itr_not_missing if nm]
        )
        n_non_nan = np.sum(not_missing)
        if n_non_nan == 0:
            return dists
        else:
            max_dist = n_non_nan * log_n_data
            return dists / max_dist

    return dists_func


def _hamming_missing_ignore_func(dists_func, weights, missing_ind):
    """Returns a wrapped Hamming distance function that ignores missing values"""
    either_missing = _either_mask_func(_missing_mask_func(missing_ind))

    @nb.jit
    def hamming(a, b):
        not_missing = ~either_missing(a, b)
        w = weights[not_missing]
        total_weight = np.sum(w)
        if total_weight == 0:
            return 1

        dists = dists_func(a, b, not_missing)
        return np.sum(dists * w) / total_weight

    return hamming


@nb.jit
def _hamming_missing_max(a, b, missing_mask):
    """
    Hamming distance function for missing values that treats them as not matching
    """
    return np.ones(np.sum(missing_mask))


def _hamming_missing_mean_func(data, missing_ind):
    """
    Returns Hamming distance function for missing values that uses the mean distance
    between non-missing values in data (per feature).
    """
    missing_mask = _missing_mask_func(missing_ind)

    # calculate mean inter-point partial distance
    one_missing_dists = []
    one_missing_values = []
    both_missing_dist = []
    for feature in data.T:
        # count each value
        f_not_missing = feature[~missing_mask(feature)]
        values, counts = np.unique(f_not_missing, return_counts=True)

        if len(values) <= 1:
            one_missing_dists.append([0])
            one_missing_values.append(values)
            both_missing_dist.append(0)
        else:
            # count cumulative pairwise distances between values
            dist_table = np.array([counts for _ in values])
            for i in range(len(dist_table)):
                dist_table[i, i] = 0

            # calculate mean distances for one-missing and both-missing cases
            sum_dist = np.sum(dist_table, axis=1)
            value_mean_dist = sum_dist / (sum_dist + counts)
            mean_dist = np.sum(value_mean_dist * counts) / np.sum(counts)

            one_missing_dists.append(value_mean_dist)
            one_missing_values.append(values)
            both_missing_dist.append(mean_dist)

    one_missing_dists = _ndarray_from_mixed_list(one_missing_dists)
    one_missing_values = _ndarray_from_mixed_list(one_missing_values)
    both_missing_dist = np.array(both_missing_dist)
    is_missing = _missing_func(missing_ind)

    @nb.jit
    def one_missing_dist(i, value):
        for j, v in enumerate(one_missing_values[i]):
            if v == value:
                return one_missing_dists[i, j]
        return 1

    @nb.jit
    def missing_dist(a, b, i):
        a_missing = is_missing(a)
        b_missing = is_missing(b)
        if a_missing and b_missing:
            return both_missing_dist[i]
        else:
            value = a if b_missing else b
            return one_missing_dist(i, value)

    @nb.jit
    def missing_dists(a, b, missing):
        itr_missing = enumerate(zip(missing, a, b))
        return np.array(
            [missing_dist(a_val, b_val, i) for i, (m, a_val, b_val) in itr_missing if m]
        )

    return missing_dists


def _hamming_missing_uniform_func(data, missing_ind):
    """
    Returns Hamming distance function for missing values that uses the mean distance
    between idealized equally-likely non-missing values (per feature) for missing
    values.
    """
    missing_mask = _missing_mask_func(missing_ind)

    # calculate mean partial distance as if values are uniformly distributed
    dists = []
    for feature in data.T:
        f_not_missing = feature[~missing_mask(feature)]
        n_values = len(np.unique(f_not_missing))
        if n_values > 0:
            dists.append((n_values - 1) / n_values)
        else:
            dists.append(0)

    dists = np.array(dists)

    @nb.jit
    def missing_dists(a, b, missing):
        return dists[missing]

    return missing_dists


########################################################################################
# public functions
########################################################################################

entropy = _shape_type_guard(_entropy)


def info_variation(a, b, missing_ind=np.nan):
    """Variation of information (distance) between a and b"""
    return _vi(np.asanyarray(a), np.asanyarray(b), missing_ind)


def hamming_variation_func(weight, dist, missing_dist, data, missing_ind=np.nan):
    """
    Returns a variation on normalized Hamming distance using any of several strategies
    for addressing missing values. This proof-of-concept implementation assumes all
    features are categorical.

    Parameters
    ----------
    weight : string
        How to weight features relative to each other. One of:
        - "uniform" - all features have equal weight
        - "entropy" - each feature is weighted by its entropy
        - "normalized entropy" - each feature is weighted by its entropy relative to
        perfect entropy for the number of unique non-missing values
    dist : string
        How to calculate partial distances for non-missing values. One of:
        - "binary" - traditional Hamming distance
        - "entropy" - distance measures smallest set containing both values
    missing_dist : string
        How to calculate partial distances for missing values. One of:
        - "ignore" - calculates as if features with missing values did not exist
        - "max" - missing values get maximum per-feature distance (1)
        - "uniform" - what mean distance for the feature would be if the non-missing
        values were uniformly distributed
        - "mean" - mean distance between feature values as they are distributed
    data : matrix
        array of data vectors to fit hamming distance function to
    missing_ind
        The indicator used in data to represent missing values
    """
    data = np.asanyarray(data)

    if weight == 'uniform':
        weights = np.ones(data.shape[1])
    elif weight == 'entropy':
        weights = np.array([_entropy(x, missing_ind) for x in data.T])
    elif weight == 'normalized entropy':
        weights = np.array([_normalized_entropy(x, missing_ind) for x in data.T])
    else:
        return None

    total_weight = np.sum(weights)
    if total_weight == 0:
        return None

    if dist == 'binary':

        @nb.jit
        def dists_func(a, b, not_missing_mask):
            return a[not_missing_mask] != b[not_missing_mask]

    elif dist == 'entropy':
        dists_func = _hamming_dist_entropy_func(data, missing_ind)
    else:
        return None

    if missing_dist == 'ignore':
        return _hamming_missing_ignore_func(dists_func, weights, missing_ind)
    elif missing_dist == 'max':
        missing_func = _hamming_missing_max
    elif missing_dist == 'mean':
        missing_func = _hamming_missing_mean_func(data, missing_ind)
    elif missing_dist == 'uniform':
        missing_func = _hamming_missing_uniform_func(data, missing_ind)
    else:
        return None

    either_missing = _either_mask_func(_missing_mask_func(missing_ind))

    @nb.jit
    def hamming(a, b):
        missing = either_missing(a, b)
        not_missing = ~missing
        dists = dists_func(a, b, not_missing) * weights[not_missing]
        dists_missing = missing_func(a, b, missing) * weights[missing]
        return (np.sum(dists) + np.sum(dists_missing)) / total_weight

    return hamming


def mutual_info(a, b, missing_ind=np.nan):
    """Mutual information between a and b"""
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    ent = fn.partial(_entropy, missing_ind=missing_ind)
    return max(0, (ent(a) + ent(b) - _vi(a, b, missing_ind)) / 2)


normalized_entropy = _shape_type_guard(_normalized_entropy)
