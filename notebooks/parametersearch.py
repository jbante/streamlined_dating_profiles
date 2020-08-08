# proof of concept for parameter searches to pair with unsupervisedcv.py
import functools as fn
import itertools as itr
import time

import joblib as jl
import multiprocess as mp
import numpy as np
import numpy.random as rand
import scipy.stats as st

import utility as utl

DEFAULT_RNG = rand.default_rng()
PHI = 1.618033988749895  # (1 + np.sqrt(5)) / 2  # golden ratio
LITTLE_PHI = 1 - 1 / PHI


def _adjusted_probs(probs, n_samples=None):
    """Applies a Laplacian adjustment to a list of probabilities

    Parameters
    ----------
    probs : array_like
        Probabilities of different outcomes for the same event
    n_samples : int
        Number of sample observations probs were calculated from

    Returns
    -------
    ndarray
        Adjusted probabilites
    """
    n = len(probs)
    m = n if n_samples is None else n_samples
    laplace_adjusted = (probs * m + 1) / (m + n)
    non_nan = np.nan_to_num(laplace_adjusted, 1 / n)
    normalized = non_nan / np.sum(non_nan)
    return normalized


def _n_jobs_actual(n_jobs):
    """Actual number of jobs to execute in parallel"""
    try:
        result = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
    except:
        result = 1
    return n_jobs


def _param_sample(spec, rng):
    """
    Random value for a single parameter from a specification.

    Parameters
    ----------
    spec : dict
    rng : numpy.random.Generator

    Returns
    -------
    dict
    """
    if spec['type'] == 'constant':
        return spec['value']

    values = spec['values']
    if spec['type'] == 'categorical':
        return rng.choice(values)

    pair = sorted(rng.choice(values, 2, replace=False))
    if spec['type'] == 'float':
        return rng.uniform(*pair)
    elif spec['type'] == 'int':
        return rng.integers(*pair, endpoint=True)
    else:
        return None


def _random_cat_param_list(vals, probs, n, rng):
    """Random list of categorical parameter values"""
    return rng.choice(vals, n, p=probs)


def _random_float_param_list(vals, probs, n, rng):
    """Random list of float parameter values"""
    pairs = [sorted(rng.choice(vals, 2, replace=False, p=probs)) for _ in range(n)]
    return [rng.uniform(*p) for p in pairs]


def _random_int_param_list(vals, probs, n, rng):
    """Random list of integer parameter values"""
    pairs = [sorted(rng.choice(vals, 2, replace=False, p=probs)) for _ in range(n)]
    result = [rng.integers(*p, endpoint=True) for p in pairs]
    return result


def _random_param_list(spec, n, rng):
    """
    Realize a random parameter from a specification

    Parameters
    ----------
    spec : dict
        param_spec dictionary containing 'type' and 'value' or 'values'
    n : int
        Number of values to create
    rng : numpy.random.Generator

    Returns
    -------
    list
    """
    if spec['type'] == 'constant':
        return [spec['value'] for _ in range(n)]

    # handle uninitialized spec
    spec_vals = spec['values']
    n_vals = len(spec_vals)
    counts = np.zeros(n_vals) if 'counts' not in spec else spec['counts']
    probs = np.full(n_vals, 1 / n_vals) if 'probs' not in spec else spec['probs']

    vals = []

    # prioritize values with 0 counts
    zeros_i = np.where(counts == 0)[0]
    if len(zeros_i) > 0:
        vals = spec_vals[zeros_i]

    n_left = n - len(vals)
    if n_left > 0:
        if spec['type'] == 'categorical':
            vals = np.append(
                vals, _random_cat_param_list(spec_vals, probs, n_left, rng)
            )
        elif spec['type'] == 'float':
            vals = np.append(
                vals, _random_float_param_list(spec_vals, probs, n_left, rng)
            )
        elif spec['type'] == 'int':
            vals = np.append(
                vals, _random_int_param_list(spec_vals, probs, n_left, rng)
            ).astype(int)

    rng.shuffle(vals)
    if len(vals) > n:
        vals = vals[:n]
    return vals


def _random_params_list(param_specs, n, rng):
    """
    Realize a list of random parameter dictionaries from a specification

    Parameters
    ----------
    spec : dict
        param_spec dictionary containing 'type' and 'value' or 'values'
    n : int
        Number of parameter combinations to create
    rng : numpy.random.Generator

    Returns
    -------
    list of dict
    """
    val_lists = [_random_param_list(spec, n, rng) for spec in param_specs.values()]
    return [
        {k: v for k, v in zip(param_specs.keys(), vals)} for vals in zip(*val_lists)
    ]


def _safe_score_func(score_func):
    """Wrapper function that returns np.NINF if score_func raises an exception"""

    def f(**kwargs):
        try:
            return score_func(**kwargs)
        except:
            return np.NINF

    return f


def _sample_best_params(param_sampler, score_func, best_params, best_score):
    """Scores parameters and returns best results vs previous best"""
    params = param_sampler()
    score = score_func(**params)
    if score > best_score:
        return params, score
    else:
        return best_params, best_score


def _top_scores_within_tol(scores, n_top, tol):
    """True if the top (largest) scores are within a tolerance of each other"""
    ordered = sorted(scores, reverse=True)
    return np.abs(ordered[0] - ordered[n_top - 1]) <= tol


def _updated_param_specs(param_specs, param_history, score_history):
    """Returns param_specs with counts & probabilities updated to match the history"""
    ranks = st.rankdata(score_history)
    prob_better = ranks / np.sum(ranks)
    new_spec = fn.partial(_updated_spec, prob_better=prob_better)
    histories = np.asanyarray(param_history).T
    zipped_specs = zip(param_specs, param_specs.values(), histories)
    return {name: new_spec(spec, history) for name, spec, history in zipped_specs}


def _updated_spec(spec, history, prob_better):
    """
    Updates a param_spec based on history of values and relative scores.

    Parameters
    ----------
    spec : dict
        A param_spec dictionary containing 'type' and 'value' or 'values'
    history : list
        Historical values of the parameter
    prob_better : list
        For each row in history, the probability that the score is better than for
        another parameter (similar to calculations for a Wilcoxon-Mann-Whitney U test)

    Returns
    -------
    dict
    """
    if spec['type'] == 'constant':
        return spec
    elif spec['type'] is 'float':
        values = np.unique(np.concatenate([spec['values'], history])).astype(float)
    elif spec['type'] is 'int':
        values = np.unique(np.concatenate([spec['values'], history])).astype(int)
    else:
        values = spec['values']

    n = len(values)
    probs = np.empty(n)
    counts = np.empty(n)
    for v, i in zip(values, range(n)):
        vals_j = history == v
        counts[i] = np.sum(vals_j)
        probs[i] = np.mean(prob_better[vals_j])

    probs = _adjusted_probs(probs, len(history))
    return {'type': spec['type'], 'values': values, 'counts': counts, 'probs': probs}


########################################################################################
# public functions
########################################################################################


def condensing_search(
    param_specs,
    score_func,
    n_iter=None,
    max_time=None,
    tol=1e-4,
    rng=DEFAULT_RNG,
    print_iter=False,
):
    """
    Randomly searches parameter values for best-scoring parameters, focusing around
    better-scoring values as they are found.

    Parameters
    ----------
    param_specs : dict
    score_func : callable
        Function that takes a realized set of parameters and returns a float. Bigger
        scores mean better parameters.
    n_iter : int
        Number of iterations of testing random parameters before returning best result
    max_time : float
        Maximum time to continue sampling scores for parameters. Actual execution time
        will usually exceed this when fitting the last sample parameters.
    tol : float
        Tolerance: exit early if the top scores are all within tol of each other
    rng : numpy.random.Generator
    print_iter : boolean
        Whether or not to print the number of completed iterations

    Returns
    -------
    dict of best-scoring parameters
    float best score
    """
    if n_iter is None and max_time is None:
        return None

    n_top = len(param_specs) ** 2
    history = []
    scores = []

    max_iter = np.inf if n_iter is None else n_iter
    end = np.inf if max_time is None else time.time() + max_time
    while len(scores) < max_iter and time.time() < end:
        params = _random_params_list(param_specs, 1, rng)[0]
        score = score_func(**params)
        history.append(list(params.values()))
        scores.append(score)
        param_specs = _updated_param_specs(param_specs, history, scores)

        if len(scores) >= n_top and _top_scores_within_tol(scores, n_top, tol):
            print("early exit")
            break

    best_i = np.argmax(scores)
    result_params = {n: v for n, v in zip(param_specs, history[best_i])}

    if print_iter:
        print(f'Condensing search iterations: {len(scores)}')
    return result_params, scores[best_i]


def param_grid(param_values):
    """
    Returns a generator of parameter dictionaries over the cartesian product of values
    in param_spec

    Parameters
    ----------
    param_values : dict
        keys are taken as parameter names, values should be values to use for each
        parameter

    Returns
    -------
    Generator
    """
    names = param_values.keys()
    value_product = itr.product(*param_values.values())
    return ({n: v for n, v in zip(names, vals)} for vals in value_product)


def param_spec_constant(x):
    return {'type': 'constant', 'value': x}


def param_spec_categorical(x):
    return {'type': 'categorical', 'values': np.asanyarray(x)}


def param_spec_float(x):
    return {'type': 'float', 'values': np.asanyarray(x)}


def param_spec_int(x):
    return {'type': 'int', 'values': np.asanyarray(x).astype(int)}


def random_param_sampler(param_specs):
    """
    Returns a function that realizes a random combination of parameter values.

    Parameters
    ----------
    param_spec : dict

    Returns
    -------
    function
    """
    names = list(param_specs.keys())
    specs = list(param_specs.values())

    def f(rng):
        return {k: _param_sample(spec, rng) for k, spec in zip(names, specs)}

    return f


def random_search(
    param_sampler, score_func, n_iter, max_time, rng=DEFAULT_RNG, print_iter=False
):
    """
    Random search for best-scoring parameter values.

    Parameters
    ----------
    param_sampler : callable
        Function that creates a random parameter dictionary to pass to score_func.
    score_func : callable
        Function that takes the result of param_sampler, and returns a float. Bigger
        scores mean better parameters.
    n_iter : int
        Number of iterations of testing random parameters before returning best
        result
    max_time : float
        Maximum time to continue sampling scores for parameters. Actual execution
        time will usually exceed this when fitting the last sample parameters.
    rng : numpy.random.Generator
    print_iter : boolean
        Whether or not to print the number of completed iterations

    Returns
    -------
    dict of best-scoring parameters
    float best score
    """
    if n_iter is None and max_time is None:
        return None

    best_params = None
    best_score = np.NINF
    max_iter = np.inf if n_iter is None else n_iter
    end = np.inf if max_time is None else time.time() + max_time
    i = 0
    while i < max_iter and time.time() < end:
        params = param_sampler(rng)
        score = score_func(**params)
        if score > best_score:
            best_params = params
            best_score = score
        i += 1

    if print_iter:
        print(f'Random search iterations: {i}')
    return best_params, best_score


def split_random_search(
    param_specs,
    score_func,
    split_ratio=LITTLE_PHI,
    n_iter=None,
    max_time=None,
    rng=DEFAULT_RNG,
    print_iter=False,
):
    """Performs random_search followed by condensing_random_search

    Parameters
    ----------
    param_specs : dict
    score_func : callable
        Function that takes a realized set of parameters and returns a float. Bigger
        scores mean better parameters.
    split_ratio : float
        Proportion of test budget (n_iter and max_time) to use for the initial
        random_search. Remainder of the budget will be used for
        condensing_random_search.
    n_iter : int
        Number of iterations of testing random parameters before returning best result
    max_time : float
        Maximum time to continue sampling scores for parameters. Actual execution time
        will usually exceed this when fitting the last sample parameters.
    tol : float
        Tolerance: exit early if the top scores are all within tol of each other
    rng : numpy.random.Generator
    print_iter : boolean
        Whether or not to print the number of completed iterations

    Returns
    -------
    dict of best-scoring parameters
    float best score
    """
    ratio = max(min(split_ratio, 1), 0)
    param_sampler = random_param_sampler(param_specs)
    first_iter = None if n_iter is None else utl.iceil(n_iter * split_ratio)
    first_time = None if max_time is None else max_time * split_ratio

    best_params, _ = random_search(
        param_sampler,
        score_func,
        first_iter,
        first_time,
        rng=rng,
        print_iter=print_iter,
    )
    for p, b in zip(param_specs.values(), best_params.values()):
        if p['type'] != 'constat':
            p['values'] = np.append(p['values'], b)

    remaining_iter = None if n_iter is None else n_iter - first_iter
    remaining_time = None if max_time is None else max_time - first_time
    return condensing_search(
        param_specs,
        score_func,
        remaining_iter,
        remaining_time,
        rng=rng,
        print_iter=print_iter,
    )
