# proof of concept unsupervised cross validation functions
import functools as fn
import itertools as itr

import numpy as np

DEFAULT_RNG = np.random.default_rng()


def _all_masked_input(func):
    """Returns a wrapper function that applies all(masks) to positional arguments"""

    def f(masks, *args):
        mask = np.all(masks, axis=0)
        masked_args = [a[mask] for a in args]
        return func(*masked_args)

    return f


def _array_selector(X):
    """Returns a function for selecting masked subsets of X with a boolean array"""
    return lambda mask: X[mask]


def _dist_matrix_selector(dist_matrix):
    """
    Returns a function for selecting masked subsets of a distance matrix with a boolean
    array
    """
    return lambda mask: dist_matrix[mask].T[mask]


def _mask_aligned_lister(func, fill_value=None):
    """
    Returns a wrapper function that feeds the inner function a reduced input array and
    expands output to the original length, with values placed according to the mask.

    Parameters
    ----------
    func : function
        A function that takes an ndarray as input and returns a list
    fill_value
        Value to fill the output where the mask is False

    Returns : function
        A function that takes a mask array and ndarray as input and returns an ndarray
    """

    def f(mask, X):
        Y_it = iter(func(X(mask)))
        return np.array([Y_it.__next__() if m else fill_value for m in mask])

    return f


def agreement_cross_val(
    score_func, estimator, data, n_folds=5, dist_matrix_data=False, rng=DEFAULT_RNG,
):
    """
    Calculates scores for an unsupervised learning estimator based on agreement on
    estimates made for data on different sets of folds. This is suitable for estimators
    that do not have a one-to-one correspondence between inputs and outputs, e.g.,
    feature selection.

    Parameters
    ----------
    score_func : function
        Funtion that takes two equal-length arrays as input and returns an agreement
        score
    estimator : function
        Function that takes an array as input, and returns an array aligned with input
    data : array_like
        Data that will be split into folds and used as input to estimator
    n_folds : int
        Number of data splits to create and estimate on (minimum of 3)
    dist_matrix_data : bool
        Set to True if data represents a distance matrix
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of floats
    """
    X = _dist_matrix_selector(data) if dist_matrix_data else _array_selector(data)
    fold_masks = kfolds_masks(len(data), n_folds, rng)
    fit_masks = [np.any(f, axis=0) for f in itr.combinations(fold_masks, n_folds - 1)]
    estimates = [estimator(X(mask)) for mask in fit_masks]
    scores = [score_func(a, b) for a, b in itr.combinations(estimates, 2)]
    return np.array(scores)


def embedding_cross_val(
    score_func, estimator, data, n_folds=5, dist_matrix_data=False, rng=DEFAULT_RNG,
):
    """
    Calculates scores of data embeddings in different spaces via cross validation.
    The held-out fold is not used to evaluate the embedding. The score_func should
    compare the original data with the embedded data.

    Parameters
    ----------
    score_func : function
        Funtion that takes two equal-length arrays as input and returns an agreement
        score
    estimator : function
        Function that takes an array as input, and returns an array aligned with input
    data : array_like
        Data that will be split into folds and used as input to estimator
    n_folds : int
        Number of data splits to create and estimate on (minimum of 2)
    dist_matrix_data : bool
        Set to True if data represents a distance matrix
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of floats
    """
    X = _dist_matrix_selector(data) if dist_matrix_data else _array_selector(data)
    fold_masks = kfolds_masks(len(data), n_folds, rng)
    fit_masks = [np.any(f, axis=0) for f in itr.combinations(fold_masks, n_folds - 1)]
    def wrapped_score(a): return score_func(estimator(a), a)
    scores = [wrapped_score(X(mask)) for mask in fit_masks]
    return np.array(scores)


def kfolds_masks(n_rows, n_folds, rng=DEFAULT_RNG):
    """Returns boolean mask arrays for a random k-fold split"""
    folds = [i % n_folds for i in range(n_rows)]
    rng.shuffle(folds)
    return np.array([[f == n for f in folds] for n in range(n_folds)])


def nonoverlap_cross_val(
    score_func, fit_func, data, n_folds=5, dist_matrix_data=False, rng=DEFAULT_RNG,
):
    """
    Calculates scores for an unsupervised learning estimator based on agreement on
    predictions made for data neither estimator was fit to. This is suitable for
    estimators with a one-to-one correspondence between inputs and outputs that can
    make predictions outside the data they were fit to, e.g., synthetic features.

    Parameters
    ----------
    score_func : function
        Funtion that takes two equal-length arrays as input and returns an agreement
        score
    fit_func : function
        Function that takes an array as input, and returns a function that returns
        predictions for similar inputs—fit_func fits a prediction model
    data : array_like
        Data that will be split into folds and used as input to predictor_fitter
    n_folds : int
        Number of data splits to create and estimate on (minimum of 3)
    distance_matrix_data : bool
        Set to True if data represents a distance matrix
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of floats
    """
    X = _dist_matrix_selector(data) if dist_matrix_data else _array_selector(data)

    # get folds; rotate through list rather than doing all combinations
    fold_masks = kfolds_masks(len(data), n_folds, rng)
    fold_range = range(n_folds)
    fit_size = n_folds - 2
    fit_folds = [[(i - j) % n_folds < fit_size for i in fold_range] for j in fold_range]
    fit_masks = [np.any(fold_masks[a], axis=0) for a in fit_folds]

    # fit predictors
    predictors = [fit_func(X(mask)) for mask in fit_masks]

    # score on non-overlaps
    pred_pairs = (
        (predictors[(i + 1) % n_folds], predictors[(i + 2) % n_folds])
        for i in fold_range
    )
    scores = [score_func(a(X(m)), b(X(m))) for (a, b), m in zip(pred_pairs, fold_masks)]
    return np.array(scores)


def overlap_cross_val(
    score_func, estimator, data, n_folds=5, dist_matrix_data=False, rng=DEFAULT_RNG,
):
    """
    Calculates scores for an unsupervised learning estimator based on agreement on
    estimates made for data on overlapping input. This is suitable for estimators with
    a one-to-one correspondence between inputs and outputs, e.g., cluster models.

    Parameters
    ----------
    score_func : function
        Function that takes two equal-length arrays as input and returns an agreement
        score
    estimator : function
        FA function that takes an array as input, and returns an array aligned with
        input
    data : array_like
        Data that will be split into folds and used as input to estimator
    n_folds : int
        Number of data splits to create and estimate on (minimum of 3)
    dist_matrix_data : bool
        Set to True if data represents a distance matrix
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of floats
    """
    X = _dist_matrix_selector(data) if dist_matrix_data else _array_selector(data)

    # get folds
    fold_masks = kfolds_masks(len(data), n_folds, rng)
    fit_masks = [np.any(f, axis=0) for f in itr.combinations(fold_masks, n_folds - 1)]

    # estimate on folds
    aligned_estimator = _mask_aligned_lister(estimator)
    estimates = [aligned_estimator(mask, X) for mask in fit_masks]

    # score on overlaps
    masked_score = _all_masked_input(score_func)
    est_comb = itr.combinations(zip(fit_masks, estimates), 2)
    scores = [masked_score([am, bm], a, b) for (am, a), (bm, b) in est_comb]
    return np.array(scores)


def subfold_cross_val(
    cross_val_func, data, n_folds=5, dist_matrix_data=False, rng=DEFAULT_RNG
):
    """
    Calculates scores for an unsupervised learning estimator by performing cross
    validation limited to one fold at a time from a larger data set. This may be
    helpful for coping with execution time problems.

    Parameters
    ----------
    cross_val_func : function
        A function that takes a data set as input and returns a list of cross
        validation scores. It should not have it's own rng.
    data : array_like
        Data that will be split into folds and used as input to estimator
    n_folds : int
        The number of data splits to create and estimate on (minimum of 3)
    dist_matrix_data : bool
        Set to True if data represents a distance matrix
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of floats
    """
    X = _dist_matrix_selector(data) if dist_matrix_data else _array_selector(data)
    fold_masks = kfolds_masks(len(data), n_folds, rng)
    scores = np.array([cross_val_func(X(mask), rng=rng) for mask in fold_masks])
    n_scores = scores.shape[0] * scores.shape[1]
    return scores.reshape(n_scores)
