import numpy as np
import scipy.spatial.distance as dist
import scipy.stats as st

DEFAULT_RNG = np.random.default_rng()


def dist_corr_score(emb_distances, original_distances):
    """
    Scores embedding quality as normalized linear correlation between original and
    mapped distances
    """
    r, p = st.pearsonr(emb_distances, original_distances)
    return ((r + 1) / 2) ** 2


def embedding_score(emb_data, original_data, original_metric):
    """Scores embedding quality on a subsample of the data"""
    n = len(emb_data)
    sample_pairs = pair_indices_sample(n, max(1000, n // 2))
    emb_dist = metric_sample(emb_data, dist.euclidean, sample_pairs)
    original_dist = metric_sample(original_data, original_metric, sample_pairs)
    return dist_corr_score(emb_dist, original_dist)


def embedding_matrix_score(emb_data, original_dist_matrix):
    """
    Scores embedding quality on a subsample of the data, using raw embedded data and
    original distance data
    """
    n = len(emb_data)
    sample_pairs = pair_indices_sample(n, max(1000, n // 2))
    emb_dist = metric_sample(emb_data, dist.euclidean, sample_pairs)
    original_dist = [original_dist_matrix[i, j] for i, j in sample_pairs]
    return dist_corr_score(emb_dist, original_dist)


def metric_sample(data, metric, sample_pairs):
    """metric evaluated on sample pairs from data"""
    return [metric(data[i], data[j]) for i, j in sample_pairs]


def pair_indices_sample(n_source, n_pairs, rng=DEFAULT_RNG):
    """Indices for a subsample of pairwise comparisons between list elements"""
    max_pairs = (n_source - 1) * (n_source - 2) / 2
    if max_pairs <= n_pairs:
        # all possible pairs
        return [[a, b] for a, b in itr.combinations(range(n_source), 2)]
    elif n_source < n_pairs:
        # sample pairs with full replacement
        return rng.choice(n_source, n_pairs * 2).reshape((n_pairs, 2))
    elif n_source < n_pairs * 2:
        # sample pairs with replacement between sides
        a_sample = rng.choice(n_source, n_pairs, replace=False)
        b_sample = rng.choice(n_source, n_pairs, replace=False)
        return [[a, b] for a, b in zip(a_sample, b_sample)]
    else:
        # sample pairs without replacement
        return rng.choice(n_source, n_pairs * 2, replace=False).reshape((n_pairs, 2))
